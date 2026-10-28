"""Single source of truth for available embedding methods.

WHY THIS MODULE EXISTS
----------------------
There are two ways to add an embedding method to this project:

1. add a branch to a dispatch function, or
2. declare it once in a table.

The refactor already established that (1) rots: the skills layer used to repeat
its key -> implementation mapping in five places, so adding one skill meant
editing five tables and forgetting one produced a *runtime* failure.  The
embedding layer therefore starts with the same discipline: one table, one
declaration per method, everything else derived.

Adding a method is exactly two edits:

1. append one :class:`AdapterRegistration` to :data:`ADAPTERS`;
2. write the adapter module it points at.

Nothing else in the package needs to change -- :func:`available`,
:func:`manifest`, :func:`entry`, and the ``models.embed`` dispatcher all read
this table.

WHY MANIFESTS ARE DECLARED SEPARATELY FROM ADAPTERS
---------------------------------------------------
A manifest is introspectable *without importing the framework*.  Listing what
this build can do must not import TensorFlow or torch (seconds of start-up, and
on some machines a hard crash).  So the table holds manifests built here, and
the adapter module -- which does import a framework -- is resolved lazily, only
when a call is actually made.
"""
from __future__ import annotations

import os

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from cellpaint_pipeline.models.errors import UnknownModel
from cellpaint_pipeline.models.contract import (
    ChannelSpec,
    InputUnit,
    ModelManifest,
    PreprocessSpec,
    WeightsSpec,
)

__all__ = [
    'ADAPTERS',
    'AdapterRegistration',
    'PENDING',
    'SURROGATES',
    'active_profile',
    'available',
    'declared',
    'describe',
    'entry',
    'manifest',
    'manifests',
    'pending',
    'profile_names',
    'register',
    'set_profile',
]


@dataclass(frozen=True)
class AdapterRegistration:
    """One embedding method: its contract, and where its code lives.

    ``module``/``factory`` are resolved lazily by :func:`entry` so that merely
    listing methods never imports a deep-learning framework.
    """

    manifest: ModelManifest
    module: str
    factory: str

    @property
    def name(self) -> str:
        return self.manifest.name

    def load_factory(self) -> Callable[..., Any]:
        from importlib import import_module

        module = import_module(self.module)
        return getattr(module, self.factory)

    @property
    def requires_frameworks(self) -> tuple[str, ...]:
        """Framework modules the **adapter that will actually run** needs.

        WHY THIS EXISTS
        ---------------
        :attr:`ModelManifest.requires` describes the *official* implementation --
        it is part of the published contract, and it is what a reader wants to
        know when the official weights are finally obtained.  But under the
        ``v3`` profile the adapter that runs is the **surrogate**, and a
        surrogate needs a strictly smaller set: the official DINO4Cells code
        imports ``kornia`` and ``oyaml``, while the surrogate is plain torch.

        ``probe()`` used to read ``manifest.requires``, so it reported
        ``dinocells`` as needing ``oyaml`` -- and therefore as *not runnable* --
        on a machine where ``models.embed('dinocells', ...)`` succeeded.  Two
        different questions were being answered by one field, and the
        diagnostics were wrong in the direction that misleads most: they told
        the operator the method was broken.

        So the question "what does *this* adapter need?" is asked of the
        adapter.  ``requires_frameworks`` is a **class attribute** and every
        adapter module imports its framework lazily inside the methods, so
        reading it costs one framework-free module import and never pulls in
        torch or TensorFlow.

        Falls back to ``manifest.requires`` when the factory has no such
        attribute -- notably the ``unavailable`` factory used for the
        cancelled methods, whose contract is still the official one.
        """
        try:
            factory = self.load_factory()
        except Exception:  # noqa: BLE001 - a missing adapter must not break probe
            return self.manifest.requires
        return tuple(
            getattr(factory, 'requires_frameworks', self.manifest.requires)
        )


# ---------------------------------------------------------------------------
# The table.
#
# Three methods have weights on this machine today.  The remaining four are
# declared so they are discoverable, and so their contracts can be validated,
# but their adapters raise MissingDependency / WeightsUnavailable rather than
# pretending to work.  Declaring a method that cannot run is deliberate: the
# caller learns *what* is missing from a typed error instead of a bare
# ImportError, and the contract stays exercisable in tests.
# ---------------------------------------------------------------------------

#: Channel names, in the order the Cell Painting CNN v1 was TRAINED on.
#:
#: CORRECTION.  This was `('DNA', 'RNA', 'ER', 'AGP', 'Mito')`, taken from the
#: vendored DeepProfiler project builder in this repository.  That source is
#: itself wrong: the upstream repository states the training order twice, and
#: both disagree with it.
#:
#:   upstream README.md:74 -- "trained on 5-channel Cell Painting images
#:                            (DNA, ER, RNA, AGP, Mito)"
#:   upstream tests/deepprofiler/profiling/test_cell_painting_cnn_v1.py:74 --
#:                            CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]
#:
#: The two orders differ by a swap of RNA and ER, so a tensor built with the old
#: constant fed the model its second and third channels transposed.  That never
#: raised: the model ran and produced plausible embeddings from mislabelled
#: inputs, which is exactly the silent failure this layer exists to prevent.
#:
#: Consequence for the earlier parity work: it compared the layer against this
#: project's own demo pipeline, and that pipeline used the same wrong order.
#: Parity therefore proved "layer == local demo", not
#: "layer == upstream training semantics".
DEEPPROFILER_CHANNELS = ('DNA', 'ER', 'RNA', 'AGP', 'Mito')

#: CellCLIP's own channel order, deliberately NOT sharing the constant above.
#:
#: Upstream `preprocessing/preprocess_images_jumpcp.py:79-83` documents the
#: JUMP-CP channel semantics as w1=DNA, w2=ER, w3=RNA, w4=AGP, w5=Mito.  It
#: happens to match DeepProfiler today, but the two are independent facts from
#: different projects; aliasing them means a future correction to one silently
#: changes the other.
CELLCLIP_CHANNELS = ('DNA', 'ER', 'RNA', 'AGP', 'Mito')

_ALL_REGISTRATIONS: tuple[AdapterRegistration, ...] = (
    # ---------------------------------------------------------------- ready --
    AdapterRegistration(
        manifest=ModelManifest(
            name='deepprofiler',
            version='cell-painting-cnn-v1',
            input_unit=InputUnit.SINGLE_CELL_CROP,
            channels=ChannelSpec(names=DEEPPROFILER_CHANNELS, order_matters=True),
            preprocess=PreprocessSpec(
                # MEASURED, and the correction of a real defect.  This was
                # declared 'raw' on the reasoning that the Keras graph has no
                # normalisation layer.  That reasoning was incomplete: the *crop
                # graph* normalises, per crop, before the model is reached
                # (`imaging/cropping.py:65-67`).  Driving the official crop graph
                # directly returned values spanning exactly [0.0000, 1.0000].
                #
                # Why the old declaration was dangerous rather than merely
                # imprecise: with scale='raw' and value_range=None,
                # `check_compatible` accepted 16-bit input without comment, which
                # is precisely the silent-wrong-embedding case this field exists
                # to catch -- measured at up to 8.8e4 absolute error.
                scale='per_crop_minmax',
                value_range=(0.0, 1.0),
                to_dtype='float32',
                batch_layout='BHWC',
                # Executable half of the contract: `models.prepare` applies the
                # per-sample, per-channel min-max that the official crop graph
                # performs, so a 16-bit crop can be handed in directly.
                source_map='per_crop_minmax_bhwc',
                # The crop size is a *configuration* value upstream
                # (`imaging/cropping.py:125` reads
                # `config["dataset"]["locations"]["box_size"]`; the upstream test
                # fixture sets 16), not a constant of the model.  What the
                # checkpoint shipped here actually expects is 32x32 -- read
                # straight off its `input_1` placeholder shape, which the
                # framework reports verbatim when the shape is wrong.
                #
                # So this is "the size this checkpoint wants", not "the size the
                # official pipeline uses".  And because the upstream operation
                # on a crop is `crop_and_resize` -- a *crop* -- silently
                # resizing to 32x32 here would invent a preprocessing step the
                # official pipeline does not have.  `resize_policy='refuse'`
                # keeps the contract honest: a wrong-sized crop is rejected
                # with `reason='spatial-size'` instead of being quietly
                # resampled into something that merely looks right.
                resize_policy='refuse',
            ),
            weights=WeightsSpec(
                source='Cell_Painting_CNN_v1.hdf5',
                format='keras-hdf5',
                sha256='87b756f579336d606fd3d387b04bd2d422f9d410e5635a5d6b2cda7615833e73',
                host='local',
                downloadable=True,
                approx_bytes=19236600,
                # The file stores 233 layer names; the graph the adapter builds
                # has 232 -- the difference is the weightless augmentation layer.
                weight_file_layer_count=233,
                rebuilt_layer_count=232,
            ),
            output_dim=672,
            # `pool5` is a legal feature_layer choice and yields 1280, so the
            # output check must accept both rather than pin one layer.
            output_dim_alternatives=(1280,),
            feature_layer='block6a_activation',
            # The size this particular checkpoint expects.  Not an official
            # constant -- see the `resize_policy` note above.
            spatial=(32, 32),
            license='BSD-3-Clause (code) / CC-BY-4.0 (weights)',
            requires=('tensorflow', 'efficientnet'),
            ascii_path_required=False,
            parity_mode='normalized',
            parity_prerequisites=('segmentation-illumination',),
            notes=(
                'Embedding is the activation of block6a_activation reduced over '
                'the spatial axes by successive mean(axis=1); 672 is the channel '
                'count of block6a_bn, not the model output (ClassProb is 490-way). '
                'Default in this project is block6a_activation -> 672; the '
                'upstream default of pool5 -> 1280 is available via feature_layer.'
            ),
        ),
        module='cellpaint_pipeline.models.adapters.deepprofiler',
        factory='DeepProfilerAdapter',
    ),
    AdapterRegistration(
        manifest=ModelManifest(
            name='ca_mae',
            version='OpenPhenom-S/16',
            input_unit=InputUnit.SINGLE_CELL_CROP,
            channels=ChannelSpec(
                # Channel-agnostic by construction: channelwise cross-attention
                # over patch tokens, measured to accept C in {1,4,5,6,11}.  So
                # neither the labels nor the count are requirements -- the
                # architecture adapts.  `names` documents the five channels this
                # project normally feeds rather than constraining the caller.
                names=('C0', 'C1', 'C2', 'C3', 'C4'),
                order_matters=False,
                names_are_labels=False,
                channel_count_fixed=False,
                # The architecture adapts to any channel count, but the
                # *checkpoint* does not: `max_in_chans=11`
                # (`huggingface_mae.py:75`) and the positional embedding is
                # generated for exactly that many modalities
                # (`vit.py:138-143`).  A twelfth channel is a shape error
                # raised from inside the model, so the contract refuses it
                # first, with a reason code.
                max_channels=11,
            ),
            preprocess=PreprocessSpec(
                # This is the one method whose normalisation is *inside* the
                # checkpoint: Normalizer() = pixels/255 then InstanceNorm2d.
                scale='div255',
                value_range=(0.0, 255.0),
                to_dtype='uint8',
                batch_layout='BCHW',
                # `scale` says the model divides by 255 internally, which is
                # the whole of the official normalisation (`normalizer.py:7`,
                # `vit_encoder.py:57-60`: `pixels / 255.0`).  So no source map:
                # the uint8 crop goes in untouched.
                #
                # It used to be `per_crop_max255`, justified by claiming the
                # checkpoint's InstanceNorm cancels the stretch.  Measured, that
                # is half right: InstanceNorm preserves *direction* under a
                # global scale (cosine 0.9995 at x0.5) but not magnitude
                # (max|diff| 25.8% of output std).  The step was harmless here
                # only because every BBBC022 crop has per-crop max 255, i.e. the
                # stretch was an identity.  Removed so the path matches upstream
                # for inputs other than this dataset too.
                #
                # Removing it is bit-for-bit neutral on BBBC022 (verified against
                # the stored embeddings), so the v2 numbers are unaffected.
                source_map='identity',
            ),
            weights=WeightsSpec(
                # `save_pretrained` layout: the checkpoint directory is itself
                # the importable package and holds the encoder code too, so the
                # parent directory is what goes on sys.path.
                source='openphenom_pkg/openphenom/model.safetensors',
                # Named .safetensors but the magic is 50 4b 03 04 (a torch.save
                # zip); safetensors.torch.load_file fails on it.
                format='torch-save-zip',
                sha256='f6e5f1c97101331b1574c9e4b99623260191c55eea5d98e40460849c0e4c4d47',
                host='local',
                downloadable=True,
                approx_bytes=712434294,
            ),
            output_dim=384,
            # `return_channelwise=True` emits one 384-vector per input channel,
            # so 384*C is a legal width for this method too.
            output_dim_per_channel=384,
            feature_layer='encoder_pool',
            spatial=(256, 256),
            license='Recursion Non-Commercial EULA',
            requires=('torch', 'transformers', 'einops'),
            ascii_path_required=False,
            parity_mode='normalized',
            notes=(
                'Variable channel count is supported: pass channel_names matching '
                'the array you supply. return_channelwise_embeddings=True yields '
                '384*C. Embeddings are reported to need batch correction '
                '(PCA-CenterScale or TVN) before comparison.'
            ),
        ),
        module='cellpaint_pipeline.models.adapters.ca_mae',
        factory='CaMaeAdapter',
    ),
    AdapterRegistration(
        manifest=ModelManifest(
            name='cellclip',
            version='neurips2025',
            input_unit=InputUnit.CHANNEL_EMBEDDINGS,
            channels=ChannelSpec(names=CELLCLIP_CHANNELS, order_matters=True),
            preprocess=PreprocessSpec(
                # Deliberately 'none': CellCLIP does not touch pixels.  Its
                # input is 5 per-channel DINOv2-Giant vectors of width 1536.
                scale='none',
                value_range=None,
                to_dtype='float32',
                batch_layout='BCHW',
            ),
            weights=WeightsSpec(
                source='cellclip/model.safetensors',
                format='safetensors',
                sha256='66c2d32a44a3e7d68deb489047df0df7676940308683b3285b1c8d28c471b550',
                host='local',
                downloadable=True,
                approx_bytes=1799564952,
            ),
            output_dim=512,
            # DINOv2-Giant (ViT-g/14) hidden width.  Declared so the contract can
            # tell a vector stack from a pixel stack before loading anything.
            input_feature_dim=1536,
            # The chain, not just the last link: CellCLIP consumes DINOv2-Giant
            # output, so a reproducible run has to identify both models.  Hash
            # computed from the local file (4.5 GB, 4.9 s) and cross-checked
            # against the audit report's independently computed value.
            chain_weights=(
                ('facebook/dinov2-giant',
                 '917d3c470db999d32a312f8542149be91c7cbac61ee8fb4b67ae3d82b79ce21f'),
            ),
            # Human-readable summary of the output (see ModelManifest).
            feature_layer='CrossChannelFormer CLS',
            spatial=None,
            license='CC BY-NC-ND 4.0',
            requires=('torch',),
            ascii_path_required=False,
            parity_mode='normalized',
            notes=(
                'The released checkpoint consumes precomputed per-channel '
                'DINOv2-Giant embeddings (5 x 1536), NOT images. Extracting those '
                'embeddings is a separate upstream stage requiring DINOv2-Giant, '
                'which is not bundled. The text tower (BERT) is inside the '
                'checkpoint but unused for image embedding.'
            ),
        ),
        module='cellpaint_pipeline.models.adapters.cellclip',
        factory='CellClipAdapter',
    ),
    # ------------------------------------------------- declared, not runnable --
    # The three below are filtered out of ADAPTERS by _PENDING_NAMES; see the
    # block after this tuple for why and for how to restore one.
    AdapterRegistration(
        manifest=ModelManifest(
            name='dinocells',
            version='broad-2023',
            input_unit=InputUnit.SINGLE_CELL_CROP,
            channels=ChannelSpec(names=DEEPPROFILER_CHANNELS, order_matters=True),
            # THE MODEL INPUT IS A CHANNEL STACK, NOT A TAPE.
            #
            # The *disk* format of the official Cell Painting dataset is a
            # "tape": the channels sit side by side in the width dimension and a
            # mask column rides along, so a cell is stored as (H, W*(C+1)).
            # But that is a **storage** format.  The official dataset class
            # folds it back open before the model ever sees it:
            #
            #   data_utils/cellpainting_dataset.py:23-47
            #       def fold_channels(image, channel_width, mode="drop"):
            #           output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")
            #           output = output[:, :, 0:-1]        # drop the mask channel
            #   data_utils/cellpainting_dataset.py:86
            #       image = fold_channels(image, channel_width)   # mode="drop"
            #
            # and the model is built for a stack:
            #
            #   run_get_features.py:87-91
            #       vits[arch](..., in_chans=config["model"]["num_channels"])
            #
            # So the official model input is (C, H, W) with **the mask
            # discarded**, and it is batched like any other image stack
            # (`batch_size_per_gpu: 24`).
            #
            # An earlier version of this contract implemented the *inverse* of
            # the official fold and fed the resulting (H, W*(C+1)) wide single-
            # channel image to the network.  The fold was implemented
            # correctly -- which is exactly the problem: it reproduced the
            # upstream **storage** format and then passed it off as the
            # **model input** format.  The contract is what a future reader
            # would use to swap in the official weights, so it has to describe
            # the official geometry.
            #
            # `mask_mode='drop'` states the official `mode="drop"` explicitly:
            # a mask may be supplied and is discarded, never fed to the model.
            preprocess=PreprocessSpec(scale='per_image_selfnorm', to_dtype='float32',
                                      batch_layout='BCHW', input_encoding='stack',
                                      mask_mode='drop',
                                      source_map='per_sample_selfnorm'),
            weights=WeightsSpec(
                source='dino4cells/Cell_Painting_data.tar.gz',
                format='torch-pth',
                host='zenodo',
                downloadable=False,
                approx_bytes=6896284400,
            ),
            output_dim=384,
            # 384 is the ViT-S width; the ViT-B checkpoint is also released and
            # yields 768, so both must pass the output check.
            output_dim_alternatives=(768,),
            feature_layer='CLS token',
            # 96 is the official crop size: the shipped configuration crops to
            # it (`config.yaml:246` and `:415`,
            # `RandomResizedCrop: {size: 96, scale: [0.05, 0.4]}`), and the
            # model's patch grid is built from it.  Caveat, stated because it
            # matters: the configuration in the repository is the **HPA**
            # variant (`datatype: HPA`, `num_channels: 4`); the Cell Painting
            # configuration is not published, so 96 is inherited from the
            # sibling case rather than read off the CP one.
            spatial=(96, 96),
            license='CC-BY-4.0 (weights) / no licence declared (code)',
            requires=('torch', 'kornia', 'oyaml'),
            parity_mode='normalized',
            notes=(
                'The shipped config.yaml is the HPA case (num_channels: 4); the '
                'Cell Painting checkpoints appear only in README.md:211-213, so '
                'the Cell Painting channel count is not stated anywhere in the '
                'official code (this project uses 5).  The crop size 96 IS in '
                'the official configuration (config.yaml:246).  Weights 6.90 GB '
                'on Zenodo, ~10 KB/s here (~180 h).'
            ),
        ),
        module='cellpaint_pipeline.models.adapters.unavailable',
        factory='make_unavailable_adapter',
    ),
    AdapterRegistration(
        manifest=ModelManifest(
            name='unidino',
            version='csbj-2025',
            input_unit=InputUnit.SINGLE_CELL_CROP,
            channels=ChannelSpec(names=DEEPPROFILER_CHANNELS, order_matters=True),
            # CORRECTED after reading the official source: `source/image_ops.py`
            # (`img = img / (2**16 - 1)`), reached from `source/io.py`'s
            # `_convert_to_float_tensor`, so 16-bit counts *are* scaled to [0,1].
            #
            # `value_range` says what the **array the contract checks** must
            # span, and that array is the one the *model* consumes -- here
            # [0, 1], because the official pipeline divides by 2**16-1 before
            # the network sees anything.
            #
            # It used to be declared (0, 65535), i.e. the *raw* domain.  That
            # was a semantic mismatch with a real cost: on the path that does
            # NOT run `prepare` (the caller hands the model its input directly),
            # the contract checked the raw 16-bit counts -- which sit inside
            # (0, 65535) -- and let them through.  The adapter then fed
            # 0..64973 straight into a network trained on [0, 1], and the
            # embedding came out different by max|diff| = 2.98 (the same order
            # as the embedding itself) **with no error anywhere**.  The raw
            # domain is now guarded where the raw array is handled: by
            # `prepare`'s dtype and double-scaling checks.
            preprocess=PreprocessSpec(scale='div65535', value_range=(0.0, 1.0),
                                      to_dtype='float32',
                                      batch_layout='BCHW',
                                      source_map='div65535'),
            weights=WeightsSpec(
                # The file the official inference script points at:
                # `run_dino_inference.sh:2` -> `--ckpt ./checkpoints/uniDINO.pth`.
                # The checkpoint is read from `state_dict["teacher"]`
                # (`source/dino/utils.py:98-109`).
                source='unidino/uniDINO.pth',
                format='torch-pth',
                host='zenodo',
                downloadable=False,
                approx_bytes=1379562715,
            ),
            # 1920, not 384: uniDINO has no pooled mode, so the width a caller
            # actually receives for the declared five channels is 384 * 5.  The
            # bare 384 that used to sit here was the *per-channel* figure, which
            # meant a caller sizing an array from `manifest.output_dim` got a
            # buffer five times too small.  `_check_output_width` hid it by
            # accepting any multiple of `output_dim_per_channel`.
            output_dim=1920,
            # uniDINO emits one DINOv2 vector per input channel, so with a
            # different channel count the width is 384*C.
            output_dim_per_channel=384,
            feature_layer='teacher ViT-S CLS',
            spatial=(224, 224),
            # Licence conflict, recorded rather than resolved: README.md:110
            # states the weights are CC BY-NC 4.0 (which permits derivatives),
            # while GitHub's licence API reports CC-BY-NC-ND-4.0 for the repo
            # (which forbids them).  The upstream prose is quoted because it is
            # the narrower claim's source; anyone redistributing should confirm
            # with the authors, since ND would prohibit a derivative adapter.
            license='BSD-3-Clause (code, README:108) / CC BY-NC 4.0 (weights, README:110; GitHub API reports CC-BY-NC-ND-4.0)',
            requires=('torch', 'timm', 'einops', 'pytorch_lightning'),
            parity_mode='normalized',
            notes=(
                'Single-channel model; the multi-channel output is per-channel '
                'embeddings concatenated in input order (source/inference_utils.py'
                ':29-34), so the result is 384 * C and IS order-sensitive '
                '(measured: a channel permutation changes the vector by up to '
                '2.109).  The scale is declared: `source/image_ops.py:14-20` '
                'divides by 2**16-1, so the input must be uint16 -- an 8-bit '
                'crop is rejected with reason=\'value-range\' rather than '
                'silently divided into a black image.  Weights 1.38 GB on Zenodo '
                '(doi:10.5281/zenodo.14988837), ~9 KB/s here (~43 h).'
            ),
        ),
        module='cellpaint_pipeline.models.adapters.unavailable',
        factory='make_unavailable_adapter',
    ),
    AdapterRegistration(
        manifest=ModelManifest(
            name='o2vae',
            version='natcomms-2024',
            input_unit=InputUnit.SINGLE_CELL_CROP,
            channels=ChannelSpec(names=('shape',), order_matters=False),
            preprocess=PreprocessSpec(scale='none', value_range=(0.0, 1.0),
                                      to_dtype='float32', batch_layout='BCHW',
                                      # O2-VAE eats a nucleus silhouette, not
                                      # intensities: `prepare` needs a binary
                                      # mask and refuses to invent one.
                                      source_map='mask_to_silhouette'),
            weights=WeightsSpec(
                # CORRECTION: O2-VAE *does* ship pretrained weights.  They are
                # tracked in the upstream repository at
                # `wandb/pretrained_models/model_mefs.pt` (commit 4c70d58) and
                # were trained on MEF cells -- real mouse embryonic fibroblasts
                # (25,000 train / 6,598 test single cells at 128x128, see
                # data/mefs/README.md).  An earlier version of this manifest used
                # a locally self-trained 50-epoch model on the o2-mnist *digits*
                # demo, on the mistaken belief that no weights had been released.
                source='o2vae/o2vae_mefs_official.pt',
                format='torch-pth',
                host='local',
                downloadable=False,
                sha256='aa70cfa8b080c9d9989c11de316a2f7f074e8bef29366c4aaec24c807c5dfc0f',
                approx_bytes=7428429,
                # The upstream checkout sits beside the canonical weights, so a
                # `models` package is resolvable through the same root.
                repo_source='o2vae',
            ),
            # From configs/config_mefs.py:46 (`zdim=512`).  Confirmed against the
            # checkpoint: `q_net.fully_net.3.bias` is (1024,) = 2 * zdim.
            output_dim=512,
            feature_layer='z_mu',
            # configs/config_mefs.py:28,33 resize the 128x128 MEF crops to 64.
            spatial=(64, 64),
            license='MIT',
            requires=('torch', 'e2cnn'),
            # The published architecture is imported from the upstream checkout
            # rather than vendored, so the repository is a runtime precondition
            # in addition to the checkpoint.
            requires_upstream_repo=True,
            parity_mode='normalized',
            notes=(
                'Official MEF weights.  Paper: Burgess J, Nirschl JJ, '
                'Zanellati M-C, Lozano A, Cohen S, Yeung-Levy S. "Orientation-'
                'invariant autoencoders learn robust representations for shape '
                'profiling of cells and organelles." Nature Communications 15, '
                '1022 (2024), doi:10.1038/s41467-024-45362-4; code '
                'github.com/jmhb0/o2vae (Zenodo 10.5281/zenodo.10206848).  '
                '("mefs" = Mouse Embryonic Fibroblasts.)  '
                'Input is a SINGLE-CHANNEL binary cell silhouette at 64x64 in '
                '[0,1] -- O2-VAE is a shape model, so for Cell Painting it '
                'consumes a mask derived from the DNA channel rather than '
                '5-channel intensity.  That modality difference is inherent '
                'to the method, not a handling choice, and must be stated '
                'whenever it is compared against intensity models.  '
                'DOMAIN SHIFT WARNING: the official checkpoint was trained on '
                'CELL silhouettes (configs/config_mefs.py:19, '
                'data_dir=.../data/mefs/scaled/cell) and no separate nucleus '
                'checkpoint is published, so feeding it a NUCLEUS silhouette '
                'is out-of-domain inference and the embedding distribution '
                'does not match the training distribution.  Also note the '
                'paper text says "embedding dimensions d = 128" while the '
                'released checkpoint measures zdim=512 (q_net.fully_net.3.bias '
                'is (1024,)); the code and the weights are authoritative.'
            ),
        ),
        module='cellpaint_pipeline.models.adapters.o2vae',
        factory='O2VaeAdapter',
    ),
    AdapterRegistration(
        manifest=ModelManifest(
            name='morphogenie',
            version='natcomms-2025',
            input_unit=InputUnit.SINGLE_CELL_CROP,
            channels=ChannelSpec(names=('C0', 'C1', 'C2'), order_matters=True),
            preprocess=PreprocessSpec(scale='none', to_dtype='float32',
                                      batch_layout='BCHW',
                                      # Upstream centres the crop in MATLAB before
                                      # the Python side resizes it; `prepare`
                                      # refuses instead of guessing.
                                      source_map='matlab_centring'),
            weights=WeightsSpec(
                source='morphogenie/Models.7z',
                format='torch-pth',
                host='gdrive',
                downloadable=False,
            ),
            # MEASURED: the persisted embedding width is c_dim, not z_dim.
            # MorphoGenie_Test.py:67 sets c_dim=10 and :68 sets z_dim=256; only
            # c_dim reaches the embedding (dvae.py:410-411 splits the encoder
            # output at c_dim).  z_dim is the GAN generator's latent.
            output_dim=10,
            feature_layer='latent (reparameterised sample -- see notes)',
            spatial=(256, 256),
            license='GPL-3.0',
            requires=('torch',),
            # MEASURED, and the reason this field exists: MorphoGenie_Test.py:297
            # unpacks `mu_real, logvar, c_real = fvae(x, encode_only=True)` while
            # gan_training/models/dvae.py:393-396 returns `(c, mu, logvar)`.  The
            # tensor the official caller persists as the embedding is therefore
            # `c` = reparametrize(mu, logvar) (:409-413), which draws noise;
            # three calls on identical input differed by up to 2.979.  A
            # deterministic reimplementation must use the SECOND return value
            # instead.
            determinism='stochastic',
            ascii_path_required=False,
            parity_mode='normalized',
            notes=(
                'Weights only on Google Drive, unreachable here.  The official '
                'extraction path is non-deterministic (see determinism).  Its '
                'centring preprocessing is MATLAB -- and the MATLAB SOURCE IS IN '
                'THE OFFICIAL REPOSITORY, at '
                'ImagePreProcessing/MultipleCells/CPA/GetCentImages.m; what is '
                'missing here is the MATLAB runtime, Cellpose and the raw data, '
                'not the code.  Measured: the MATLAB crop is 200x200 '
                '(GetCentImages.m:26, `Crop=[Centerx-100, Centery-100, 200, 200]`) '
                'and is min-max stretched to uint8 there (:38); the 256x256 '
                'resize happens on the Python side '
                '(gan_training/utils.py:352-357, Resize + Normalize(0.5)).  So '
                'the method is not reproducible from the repository alone -- but '
                'the reason is the missing runtime and data, not missing code.  '
                'The shipped default is nc=3 (RGB); no Cell Painting example is '
                'provided upstream.'
            ),
        ),
        module='cellpaint_pipeline.models.adapters.unavailable',
        factory='make_unavailable_adapter',
    ),
)

#: Methods that are described in full but deliberately **not** registered.
#:
#: They were cancelled on 2026-10-21, not because the contracts were wrong -- the
#: manifests below are the result of reading the official sources and measuring
#: the models -- but because their weights cannot be obtained here, and a method
#: that cannot run should not appear in ``runnable()``/``declared()`` at all
#: while it only adds noise to every listing:
#:
#: ============  ==========================================================
#: dinocells     Weights are genuinely open (Zenodo 8061392, CC-BY-4.0, open
#:               access), but the only artefact is a single 3.71 GB tarball
#:               that also carries the whole HPA FOV dataset, and the
#:               checkpoint cannot be fetched alone.  Measured 2026-10-21 at
#:               9.2 KB/s sustained -> ~110 h.  Not "short".
#: unidino       Weights on Zenodo (uniDINO.pth, 1.38 GB), same rate.
#: morphogenie   Weights on Google Drive (unreachable here), and the pipeline
#:               needs a MATLAB centring step whose runtime and data are not
#:               available here (the MATLAB source IS in the official repo).
#: ============  ==========================================================
#:
#: Kept as data rather than deleted so restoring one is a one-line change:
#: drop its name from :data:`_PENDING_NAMES` and write the adapter.  The
#: contracts stay reviewable in the meantime (see :func:`pending`).
_PENDING_NAMES: frozenset[str] = frozenset({'dinocells', 'unidino', 'morphogenie'})

#: The registered, usable methods.
ADAPTERS: tuple[AdapterRegistration, ...] = tuple(
    entry_ for entry_ in _ALL_REGISTRATIONS if entry_.name not in _PENDING_NAMES
)

#: Contract-only registrations, retained for a future re-add.  Not usable.
PENDING: tuple[AdapterRegistration, ...] = tuple(
    entry_ for entry_ in _ALL_REGISTRATIONS if entry_.name in _PENDING_NAMES
)

# ---------------------------------------------------------------- profiles --

#: What a profile is
#: ----------------
#: The same code has to answer two different questions, and confusing them is how
#: a surrogate ends up in a results table next to a real method:
#:
#: ``v2``  Which methods can I run with the *authors'* weights?  Four.  This is
#:         the default, because it is the answer that can be published.
#: ``v3``  Which methods can the *dispatch layer* drive end to end -- including
#:         three whose published weights cannot be obtained here, paired with
#:         clearly-labelled surrogate checkpoints?  Seven.
#:
#: A profile is a view over the registrations, not a separate code path: the
#: adapters, the manifests and the preprocessing are the same objects.  Switching
#: only changes which names resolve.
# ------------------------------------------------------------ v3 surrogates --

#: SHA-256 of the surrogate checkpoints this repository produced.
#:
#: A surrogate is not distributed, so there is no upstream digest to compare
#: against -- but that is exactly why it needs one.  Without a pinned digest,
#: swapping the checkpoint file would change every embedding while nothing in
#: the contract noticed; the digest is checked before the weights are loaded,
#: so a swapped file is refused rather than silently served.
#:
#: **A surrogate's digest is only valid for the file that was pinned.**  Retraining
#: is NOT reproducible: VICReg's trajectory is chaotic, so a rerun with the same
#: seed and the same data lands on different weights (measured 2026-10-27: the
#: same recipe produced ``a0265184…`` instead of ``e53838fe…``, with an
#: embedding std of 0.67 instead of 0.40 -- both non-collapsed, both different).
#: So the pinned digest detects an *accidental* replacement, which is its job,
#: but a deliberate retrain requires updating this table AND regenerating the
#: v3 embeddings and scores.
#:
#: ``dinocells`` was retrained on 2026-10-26 after the input geometry changed
#: from the official *storage* format (a wide tape) to the official *model
#: input* format (a channel stack).  The digest therefore changed with it, and
#: the old checkpoint is kept at
#: ``E://cpc_weights//_backup_20261026//surrogate_dinocells_OLD_tape.pt``.
SURROGATE_SHA256: dict[str, str] = {
    'unidino': (
        'a0265184661d511e692603aefd5a20343adb1fc1545672a8c32bb95493ae9d55'
    ),
    'dinocells': (
        '5e172d8fed4876ff4951db92796eb8cc300163281c8cf7391d717801e7dc95e4'
    ),
    'morphogenie': (
        '491029978fef0969ed7ed56abaaf189faf34f6abc936c13e1487e877d41dc86d'
    ),
}


def _surrogate_manifest(base: ModelManifest, filename: str) -> ModelManifest:
    """Re-point a cancelled contract at a locally trained surrogate checkpoint.

    Everything about the *interface* stays: same input spec, same channel rules,
    same output width, same notes.  Only the provenance changes, and it changes
    in three places at once so it cannot be missed downstream -- the manifest
    flag, the weights record, and (through the flag) every ``EmbeddingResult``.
    """
    return replace(
        base,
        weights=WeightsSpec(
            source=f'{base.name}/{filename}',
            format='torch-pth',
            host='local',
            # Pinned, so a swapped checkpoint is caught before it is loaded.
            # See SURROGATE_SHA256 above.
            sha256=SURROGATE_SHA256.get(base.name),
            # Not downloadable: it is produced locally by train_surrogates.py,
            # not fetched.  Claiming otherwise would send a user looking for a
            # release that does not exist.
            downloadable=False,
        ),
        weights_provenance='surrogate',
        notes=(
            base.notes
            + "  [SURROGATE] The checkpoint loaded here was trained on this "
              "repository's demo crops by scripts/embedding_benchmark/"
              "train_surrogates.py; the authors' weights could not be "
              "obtained. The architecture and the input/output contract are "
              "the published ones, so the interface is exercised for real -- "
              "but any score is NOT this method's published performance and "
              "must never be compared against one."
        ),
    )


#: The three methods paired with surrogate checkpoints, visible only under the
#: ``v3`` profile.  Built from :data:`PENDING` so the measured contracts stay the
#: single source of truth.
SURROGATES: tuple[AdapterRegistration, ...] = (
    AdapterRegistration(
        manifest=_surrogate_manifest(
            next(r.manifest for r in PENDING if r.name == 'unidino'),
            'surrogate_unidino.pt',
        ),
        module='cellpaint_pipeline.models.adapters.unidino',
        factory='UniDinoSurrogateAdapter',
    ),
    AdapterRegistration(
        manifest=_surrogate_manifest(
            next(r.manifest for r in PENDING if r.name == 'dinocells'),
            'surrogate_dinocells.pt',
        ),
        module='cellpaint_pipeline.models.adapters.dinocells',
        factory='DinoCellsSurrogateAdapter',
    ),
    AdapterRegistration(
        manifest=_surrogate_manifest(
            # The official extraction path draws a reparameterised sample, which
            # makes it stochastic.  The surrogate returns the encoder mean
            # instead, so it is deterministic *and says so*: a benchmark that
            # expects repeated runs for a stochastic method would otherwise
            # average draws that never vary.
            replace(
                next(r.manifest for r in PENDING if r.name == 'morphogenie'),
                determinism='deterministic',
            ),
            'surrogate_morphogenie.pt',
        ),
        module='cellpaint_pipeline.models.adapters.morphogenie',
        factory='MorphoGenieSurrogateAdapter',
    ),
)


_PROFILE_NAMES: tuple[str, ...] = ('v2', 'v3')

#: Human-readable aliases.  The two release trees are called *v1* (four methods
#: with the authors' weights) and *v2* (the same four plus three surrogate-backed
#: methods), while the profile identifiers predate that naming and stayed
#: ``v2``/``v3``.  Accepting both spellings keeps either vocabulary working; it
#: does not add a third behaviour, because an alias maps onto an existing profile.
_PROFILE_ALIASES: dict[str, str] = {
    'v2': 'v2', 'official': 'v2', 'official4': 'v2', 'default': 'v2',
    'v3': 'v3', 'all': 'v3', 'all7': 'v3',
}


def normalize_profile(name: str) -> str:
    """Map a profile name or alias onto a canonical profile identifier."""
    key = str(name).strip().lower()
    if key in _PROFILE_ALIASES:
        return _PROFILE_ALIASES[key]
    raise ValueError(
        f'unknown profile {name!r}; expected one of '
        f'{sorted(_PROFILE_ALIASES)} (canonical: {list(_PROFILE_NAMES)})'
    )


#: Default profile for this tree.  A release tree ships a generated
#: ``_release_profile.py`` whose ``PROFILE`` constant decides the answer, so the
#: two trees differ by one line of configuration instead of forked source, and a
#: plain ``import models`` already reflects the right method set.
try:  # pragma: no cover - exercised by the release builds
    from ._release_profile import PROFILE as _TREE_PROFILE
except Exception:  # noqa: BLE001 - absent in the development tree
    _TREE_PROFILE = 'v2'

#: Initial profile.  Read from the environment so the two release units can
#: differ by configuration rather than by forked source: ``cpc-embedding-v1``
#: ships the four official-weight methods, ``cpc-embedding-v2`` adds the three
#: surrogates.  An unrecognised value falls back to ``'v2'`` instead of raising,
#: because failing at import time would make the package unimportable rather
#: than merely misconfigured.
try:
    _PROFILE = normalize_profile(
        os.environ.get('CPC_EMBEDDING_PROFILE', _TREE_PROFILE)
    )
except ValueError:
    # An unrecognised value must not make the package unimportable.
    _PROFILE = 'v2'

#: Name lookup for the active profile; rebuilt by :func:`set_profile`.
_BY_NAME: dict[str, AdapterRegistration] = {}


def active_profile() -> str:
    """Return the profile currently in force (``'v2'`` or ``'v3'``)."""
    return _PROFILE


def profile_names() -> list[str]:
    """Return the available profile names, in declaration order."""
    return list(_PROFILE_NAMES)


def _profile_members(profile: str | None = None) -> tuple[AdapterRegistration, ...]:
    """Return the registrations visible under ``profile`` (default: active).

    Read through ``globals()`` rather than capturing :data:`ADAPTERS` in a dict:
    :func:`register` rebinds that name at runtime, and a captured tuple would
    quietly stop seeing contributions.
    """
    chosen = _PROFILE if profile is None else profile
    base: tuple[AdapterRegistration, ...] = globals()['ADAPTERS']
    if chosen == 'v2':
        return base
    if chosen == 'v3':
        return base + SURROGATES
    raise ValueError(
        f'unknown profile {chosen!r}; expected one of {list(_PROFILE_NAMES)}'
    )


def set_profile(name: str) -> str:
    """Switch the active profile, returning the previous one.

    Raises:
        ValueError: for an unknown profile.

    Also drops the adapter instances cached by
    :mod:`~cellpaint_pipeline.models.embedding`, because the same method name
    resolves to *different weights* in the two profiles -- a cached official
    adapter reused under ``v3`` (or the reverse) would silently answer with the
    wrong checkpoint.
    """
    global _PROFILE
    name = normalize_profile(name)
    previous, _PROFILE = _PROFILE, name
    _rebuild_by_name()
    if previous != name:
        from . import embedding

        embedding.clear_adapter_cache()
    return previous


def _rebuild_by_name() -> None:
    """Rebuild the name lookup for the active profile."""
    _BY_NAME.clear()
    _BY_NAME.update({entry_.name: entry_ for entry_ in _profile_members()})


def declared() -> list[str]:
    """Return every method name that resolves under the active profile.

    "Declared" is the precise word: a name here has a manifest and a
    registration, which is enough to call :func:`describe` or :func:`manifest`,
    but **not** enough to run it.  Use ``models.runnable()`` for the subset that
    can actually embed on this machine.

    Under the default ``v2`` profile this is the four methods with the authors'
    weights.  Under ``v3`` it also includes the three surrogate-backed methods;
    check ``manifest(name).weights_provenance`` before reporting any of their
    numbers as that method's performance.
    """
    return [entry_.name for entry_ in _profile_members()]


def available() -> list[str]:
    """Deprecated alias of :func:`declared`.

    Kept because the name reads like "can I use this?", which it never answered:
    it lists every declared method, including ones that cannot run.  Callers that
    want that question answered should use ``models.runnable()`` instead.
    """
    return declared()


def manifests() -> list[ModelManifest]:
    """Return the manifest of every method that resolves, in declaration order."""
    return [entry_.manifest for entry_ in _profile_members()]


def pending() -> list[ModelManifest]:
    """Return the manifests of the cancelled, unregistered methods.

    These are contracts without a registration: they cannot be embedded from
    here, and :func:`manifest` does not resolve them.  The list exists so the
    reasons stay checkable and restoring one later does not mean re-reading the
    upstream sources from scratch.
    """
    return [entry_.manifest for entry_ in PENDING]


def entry(name: str) -> AdapterRegistration:
    """Return the registration for ``name``.

    Raises:
        UnknownModel: when the name is not declared.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise UnknownModel(name, tuple(_BY_NAME)) from None


def manifest(name: str) -> ModelManifest:
    """Return the manifest for ``name`` without importing any framework."""
    return entry(name).manifest


def describe(name: str) -> dict:
    """Return a JSON-serialisable description of one method."""
    reg = entry(name)
    m = reg.manifest
    return {
        'name': m.name,
        'version': m.version,
        'input_unit': m.input_unit.value,
        'channels': list(m.channel_names),
        'channel_order_matters': m.channels.order_matters,
        'preprocess': {
            'scale': m.preprocess.scale,
            'value_range': list(m.preprocess.value_range) if m.preprocess.value_range else None,
            'to_dtype': m.preprocess.to_dtype,
            'batch_layout': m.preprocess.batch_layout,
            'illum_corrected': m.preprocess.illum_corrected,
        },
        'spatial': list(m.spatial) if m.spatial else None,
        'output_dim': m.output_dim,
        'feature_layer': m.feature_layer,
        'license': m.license,
        'requires': list(m.requires),
        'ascii_path_required': m.ascii_path_required,
        'parity_mode': m.parity_mode,
        # 'official' or 'surrogate'.  In the listing, not only in the manifest:
        # anything that renders a method table from describe() would otherwise
        # show a surrogate beside a real method with no visible difference.
        'weights_provenance': m.weights_provenance,
        'weights': {
            'source': m.weights.source,
            'format': m.weights.format,
            'host': m.weights.host,
            'downloadable': m.weights.downloadable,
            'approx_bytes': m.weights.approx_bytes,
            'sha256': m.weights.sha256,
        },
        'adapter': f'{reg.module}:{reg.factory}',
        'notes': m.notes,
    }


def register(registration: AdapterRegistration, *, replace: bool = False) -> None:
    """Add a method at runtime.

    Exposed so downstream users can contribute a method without forking the
    package.  ``replace=False`` refuses to shadow an existing name, which keeps
    the built-in contracts authoritative by default.
    """
    if not replace and registration.name in _BY_NAME:
        raise ValueError(
            f'embedding method {registration.name!r} is already registered; '
            f'pass replace=True to override'
        )
    ADAPTERS_GLOBAL = globals()['ADAPTERS']
    if replace and registration.name in _BY_NAME:
        existing = _BY_NAME[registration.name]
        index = list(ADAPTERS_GLOBAL).index(existing)
        as_list = list(ADAPTERS_GLOBAL)
        as_list[index] = registration
        globals()['ADAPTERS'] = tuple(as_list)
    else:
        globals()['ADAPTERS'] = tuple(ADAPTERS_GLOBAL) + (registration,)
    _rebuild_by_name()


# Populate the lookup once at import; `set_profile` and `register` keep it current.
_rebuild_by_name()
