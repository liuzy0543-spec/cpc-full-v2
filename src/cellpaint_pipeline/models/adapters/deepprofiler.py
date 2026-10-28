"""DeepProfiler Cell Painting CNN v1 adapter.

WHAT THIS METHOD PRODUCES
-------------------------
The published checkpoint is an EfficientNet-B0 trained on 5-channel Cell
Painting single-cell crops.  The embedding is **not** the model output: the
network ends in a 490-way ``ClassProb`` softmax, while the embedding is an
intermediate activation.  This project's default is ``block6a_activation``,
whose channel dimension is 672, reduced over the spatial axes by successive
``mean(axis=1)`` -- the same reduction the upstream profiler performs.

Why 672 and not the model output: measured.  ``block6a_bn`` stores four tensors
of shape ``(672,)``; the spatial extent at box_size 32 is 1x1, at 64 it is 2x2,
at 128 it is 4x4, and the reduction always lands on 672.

WHAT IS EASY TO GET WRONG (and is therefore checked, not assumed)
----------------------------------------------------------------
1. **Input scale.**  The Keras graph itself has no normalisation layer, but the
   *crop graph* does: it min-maxes each crop onto [0, 1] before the network sees
   it (``imaging/cropping.py:65-67``).  Driving the official crop graph returned
   values spanning exactly [0.0000, 1.0000].  So the manifest declares
   ``scale='per_crop_minmax'`` with ``value_range=(0.0, 1.0)``.  An earlier
   version declared ``scale='raw'`` on the reasoning that "the graph has no
   normalisation layer"; that was wrong and dangerous, because it let callers
   pass 16-bit values with no warning -- measured at up to 4.5e6 absolute error.
2. **Channel order.**  The model was trained on
   ``(DNA, ER, RNA, AGP, Mito)`` -- stated in upstream ``README.md:74`` and in
   ``tests/deepprofiler/profiling/test_cell_painting_cnn_v1.py:74``.  This
   repository's own pipeline builder used ``(DNA, RNA, ER, AGP, Mito)``, which
   transposes RNA and ER; the contract constant was corrected to the upstream
   order and now rejects the old one.  Swapping any two channels changes the
   embedding silently, so the contract check refuses a mismatched order.
3. **The augmentation layer.**  The checkpoint stores 233 layer names including
   ``augmentation_layer_1``; the graph built here has 232 and does *not* contain
   that layer, so calling ``get_layer('augmentation_layer_1')`` raises.  Measured
   behaviour is fully deterministic (identical forwards are bitwise equal), so no
   eval-mode switch is needed or wanted.

FEATURE LAYER
-------------
``feature_layer`` is a constructor argument because the same checkpoint exposes
two documented embeddings -- ``block6a_activation`` (672, this project's
pipeline output) and ``pool5`` (1280, the upstream README default).  Keeping
both reachable lets a later run compare them without editing the registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cellpaint_pipeline.models.adapters.base import BaseAdapter
from cellpaint_pipeline.models.contract import ModelManifest
from cellpaint_pipeline.models.errors import EmbeddingError

__all__ = ['DeepProfilerAdapter']

#: Training-head width baked into the checkpoint; its weights are skipped.
_CLASS_PROB_WIDTH = 490


@dataclass
class DeepProfilerAdapter(BaseAdapter):
    """Adapter for the Cell Painting CNN v1 checkpoint."""

    manifest: ModelManifest = None  # type: ignore[assignment]
    #: Activation to read.  ``None`` -> the manifest's declared layer.
    feature_layer: str | None = None
    batch_size: int = 64
    #: Extent used only for the placeholder; the graph is fully convolutional,
    #: so any spatial size is accepted at call time.
    probe_box: int = 32
    #: Requested compute device (``None`` -> the adapter decides).  Declared as a
    #: field so the generated ``__init__`` accepts it instead of shadowing
    #: :meth:`BaseAdapter.__init__`, which is what made ``device=`` a TypeError.
    device: str | None = None
    #: Additional roots searched for the checkpoint before the defaults.
    weights_roots: tuple[Any, ...] | None = None

    #: Framework modules, general-to-specific.
    requires_frameworks: tuple[str, ...] = ('tensorflow', 'efficientnet')
    dependency_hint: str | None = (
        'The DeepProfiler stack (tensorflow 2.x + efficientnet 1.1.1) is not '
        'installed in this interpreter; run with the environment that has it.'
    )

    def __post_init__(self) -> None:
        if self.manifest is None:  # only when constructed directly, not via registry
            from cellpaint_pipeline.models.registry import manifest as manifest_for

            self.manifest = manifest_for('deepprofiler')
        BaseAdapter.__init__(
            self,
            manifest=self.manifest,
            weights_roots=self.weights_roots,
            device=self.device,
        )
        if self.feature_layer is None:
            self.feature_layer = self.manifest.feature_layer
        self._graph_metadata: dict[str, Any] = {}

    # ------------------------------------------------------------ protocol --

    def check_dependencies(self) -> None:
        self.require_frameworks()

    def _build(self) -> Any:
        import os

        import numpy as np

        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        import efficientnet.tfkeras as efn
        import tensorflow as tf

        checkpoint = self.resolve_weights()

        # Upstream runs in TF1 graph mode; the vendored profiler calls this at
        # import time and the checkpoint was produced under it.
        tf.compat.v1.disable_v2_behavior()

        channels = len(self.manifest.channel_names)
        inputs = tf.keras.layers.Input((self.probe_box, self.probe_box, channels))
        base = efn.EfficientNetB0(input_tensor=inputs, include_top=False, weights=None)
        pooled = tf.keras.layers.GlobalAveragePooling2D(name='pool5')(
            base.layers[-1].output
        )
        head = tf.keras.layers.Dense(
            _CLASS_PROB_WIDTH, activation='softmax', name='ClassProb'
        )(pooled)
        model = tf.keras.Model(inputs=inputs, outputs=[head])

        # Upstream renames the head before loading so the checkpoint's own Dense
        # head is skipped and only backbone weights are transferred.
        model.layers[-1]._name = 'classifier'

        # `load_weights(by_name=True)` returns nothing and silently skips any
        # layer whose name does not match.  If every name failed to match, the
        # model would stay randomly initialised and still produce plausible
        # embeddings -- the exact failure mode this check exists to catch.  So
        # one reference tensor is snapshotted first and the load is required to
        # have changed it.  The reference must be the first layer that actually
        # *has* weights: `base.layers[0]` is an InputLayer with none.
        reference_layer = next(
            (layer for layer in base.layers if layer.get_weights()), None
        )
        if reference_layer is None:
            raise EmbeddingError(
                'the rebuilt graph exposes no layer with weights, so weight '
                'loading cannot be verified',
                reason='weight-load-unverifiable',
            )
        before = [np.array(w) for w in reference_layer.get_weights()]
        model.load_weights(str(checkpoint), by_name=True)
        after = [np.array(w) for w in reference_layer.get_weights()]

        if all(np.array_equal(old, new) for old, new in zip(before, after)):
            raise EmbeddingError(
                f'checkpoint weights did not reach the graph: reference layer '
                f'{reference_layer.name!r} is byte-identical before and after '
                f'load_weights(by_name=True), which means no layer name matched',
                reason='weight-load-noop',
                details=[f'checkpoint: {checkpoint}'],
            )

        if not any(layer.name == self.feature_layer for layer in model.layers):
            raise EmbeddingError(
                f'feature layer {self.feature_layer!r} is not present in the '
                f'rebuilt graph; the checkpoint and this builder disagree',
                reason='feature-layer-missing',
                details=[f'graph has {len(model.layers)} layers'],
            )

        self._graph_metadata = {
            'rebuilt_layer_count': len(model.layers),
            'weight_file_layer_count': self.manifest.weights.weight_file_layer_count,
            'weights_changed_first_layer': True,
            'probe_box': self.probe_box,
        }
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer(self.feature_layer).output,
        )

    def _forward(self, array: Any, **kwargs: Any) -> Any:
        import numpy as np

        batch_size = int(kwargs.get('batch_size', self.batch_size))
        features = self._model.predict(
            np.asarray(array), batch_size=batch_size, verbose=0
        )
        # Upstream reduces spatial extent by repeated mean over axis 1; kept
        # verbatim so the reduction order (and the float result) matches.
        reduced = np.asarray(features)
        while reduced.ndim > 2:
            reduced = np.mean(reduced, axis=1)
        return reduced
