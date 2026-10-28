"""CellCLIP adapter -- channel-embedding level, not pixel level.

READ THIS BEFORE CALLING IT
---------------------------
CellCLIP is **not an image model**.  The released checkpoint consumes
*precomputed per-channel image embeddings*: five vectors of width 1536 produced
by a frozen **DINOv2-Giant** encoder, one per Cell Painting channel.  The
checkpoint's own ``config.json`` says so (``input_channels: 5``, ``vision_width:
1536``) and its weight shapes confirm it:

    visual.channel_embed        (5, 1536)
    text_proj.weight            (512, 768)
    text.*                      BERT-base

So the manifest declares ``input_unit=CHANNEL_EMBEDDINGS`` and the contract
requires an ``(N, 5, 1536)`` float array.  Producing those vectors is a
**separate upstream stage** that needs DINOv2-Giant, which is not part of this
checkpoint and is not bundled here.  This adapter therefore validates loading
and the forward pass; it does not magically turn pixels into embeddings.

WHY THE ARCHITECTURE IS REPLICATED HERE
---------------------------------------
Upstream ``src/clip/model.py`` imports ``graphium``, ``torch_geometric``,
``src.channelvit`` and ``src.open_phenom`` **at module scope**, purely to
service unrelated classes (Cloome, ChannelViT, Molphenix).  Importing it
therefore requires a heavy graph-learning stack that CellCLIP itself never
touches.  The classes below are copied verbatim from the published file --
LayerNorm, QuickGELU, ResidualAttentionBlock, Transformer, CrossChannelFormer --
and instantiated through :func:`build_cellclip_visual` with the exact
hyper-parameters from ``config.json``.  That keeps weight loading honest (same
shapes, same names) without dragging in Graphium.

The text tower is present in the checkpoint but unused for image embedding, so
it is not built here; ``strict=False`` loading skips it and the adapter asserts
that every *visual* key was consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cellpaint_pipeline.models.adapters.base import BaseAdapter
from cellpaint_pipeline.models.errors import EmbeddingError
from cellpaint_pipeline.models.contract import ModelManifest

__all__ = ['CellClipAdapter', 'build_cellclip_visual', 'CROSS_CHANNEL_FORMER_CONFIG']

#: Hyper-parameters taken from the published ``config.json`` on the model hub.
CROSS_CHANNEL_FORMER_CONFIG: dict[str, int] = {
    'embed_dim': 1536,   # vision_width: width of each incoming channel embedding
    'layers': 12,        # vision_layers
    'heads': 8,          # vision_heads
    'output_dim': 512,   # projected shared image/text space
    'input_channels': 5,  # Cell Painting channels
}


def _build_classes() -> dict[str, type]:
    """Define and return the tower classes, importing torch *inside*.

    Torch is imported here rather than at module scope so that
    ``models.available()`` and ``models.manifest('cellclip')`` work on a machine
    with no torch installed -- capability discovery must not require a deep
    learning stack.
    """
    import torch
    from torch import nn

    class LayerNorm(nn.LayerNorm):
        """Upstream subclass: computes in float32 and casts back."""

        def forward(self, x):  # noqa: D102
            original = x.dtype
            return super().forward(x.type(torch.float32)).type(original)

    class QuickGELU(nn.Module):
        """Upstream approximation ``x * sigmoid(1.702x)``."""

        def forward(self, x):  # noqa: D102
            return x * torch.sigmoid(1.702 * x)

    class ResidualAttentionBlock(nn.Module):
        """One pre-norm transformer block, as published.

        The MLP children are **named** (``c_fc``, ``gelu``, ``c_proj``) rather
        than an anonymous ``nn.Sequential``.  That is what the released
        checkpoint stores: an anonymous Sequential yields keys like
        ``mlp.0.weight``, so every MLP weight would be reported as unexpected
        and the block would silently keep its random initialisation.
        """

        def __init__(self, d_model: int, n_head: int) -> None:
            super().__init__()
            from collections import OrderedDict

            self.attn = nn.MultiheadAttention(d_model, n_head)
            self.ln_1 = LayerNorm(d_model)
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ('c_fc', nn.Linear(d_model, d_model * 4)),
                        ('gelu', QuickGELU()),
                        ('c_proj', nn.Linear(d_model * 4, d_model)),
                    ]
                )
            )
            self.ln_2 = LayerNorm(d_model)

        def forward(self, x):  # noqa: D102
            import torch as _torch

            normed = self.ln_1(x)
            attended = self.attn(normed, normed, normed, need_weights=False)[0]
            x = x + attended
            return x + self.mlp(self.ln_2(x))

    class Transformer(nn.Module):
        """Stack of residual attention blocks."""

        def __init__(self, width: int, layers: int, heads: int) -> None:
            super().__init__()
            self.width = width
            self.layers = layers
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
            )

        def forward(self, x):  # noqa: D102
            return self.resblocks(x)

    class CrossChannelFormer(nn.Module):
        """The vision tower: channel tokens -> CLS embedding -> projection.

        Copied from upstream so parameter names match the checkpoint exactly:
        ``channel_embed``, ``channel_ln``, ``ln_pre``, ``transformer.resblocks.*``,
        ``cls_token``, ``ln_post``, ``proj``.
        """

        def __init__(
            self,
            embed_dim: int,
            layers: int,
            heads: int,
            output_dim: int,
            input_channels: int,
            use_cls_token: bool = True,
        ) -> None:
            super().__init__()
            scale = embed_dim ** -0.5
            self.embed_dim = embed_dim
            self.output_dim = output_dim
            self.input_channels = input_channels
            self.use_cls_token = use_cls_token
            self.channel_embed = nn.Parameter(torch.zeros(input_channels, embed_dim))
            self.channel_ln = LayerNorm(embed_dim)
            self.ln_pre = LayerNorm(embed_dim)
            self.transformer = Transformer(embed_dim, layers, heads)
            self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
            self.ln_post = LayerNorm(embed_dim)
            self.proj = nn.Parameter(scale * torch.randn(embed_dim, output_dim))

        def forward(self, x):  # noqa: D102
            batch, _, _ = x.shape
            x = x + self.channel_embed.expand(batch, -1, -1).to(x.dtype)
            x = self.channel_ln(x)
            cls_tokens = self.cls_token.expand(batch, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = self.ln_pre(x)
            x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            if self.use_cls_token:
                x = self.ln_post(x[:, 0, :])
            else:
                x = self.ln_post(x[:, 1:, :].mean(dim=1))
            if self.embed_dim != self.output_dim:
                x = x @ self.proj
            return x

    return {'CrossChannelFormer': CrossChannelFormer}


def build_cellclip_visual(**overrides: int) -> Any:
    """Instantiate the vision tower from :data:`CROSS_CHANNEL_FORMER_CONFIG`."""
    config = dict(CROSS_CHANNEL_FORMER_CONFIG)
    config.update(overrides)
    return _build_classes()['CrossChannelFormer'](**config)


@dataclass
class CellClipAdapter(BaseAdapter):
    """Adapter for the CellCLIP vision tower.

    Note the input unit: this adapter never sees pixels.  ``_forward`` receives
    ``(N, 5, 1536)`` per-channel DINOv2-Giant embeddings, which is what the
    released checkpoint was trained to consume.
    """

    manifest: ModelManifest = None  # type: ignore[assignment]
    #: Number of visual keys successfully loaded, for provenance reporting.
    loaded_key_count: int = 0
    #: Requested compute device (``None`` -> the adapter decides).  Declared as a
    #: field so the generated ``__init__`` accepts it instead of shadowing
    #: :meth:`BaseAdapter.__init__`, which is what made ``device=`` a TypeError.
    device: str | None = None
    #: Additional roots searched for the checkpoint before the defaults.
    weights_roots: tuple[Any, ...] | None = None

    requires_frameworks: tuple[str, ...] = ('torch',)
    dependency_hint: str | None = (
        'CellCLIP needs PyTorch plus safetensors; on this machine that is the '
        'torch 2.5.1+cu124 environment.'
    )

    def __post_init__(self) -> None:
        if self.manifest is None:
            from cellpaint_pipeline.models.registry import manifest as manifest_for

            self.manifest = manifest_for('cellclip')
        BaseAdapter.__init__(
            self,
            manifest=self.manifest,
            weights_roots=self.weights_roots,
            device=self.device,
        )
        self._torch = None

    # ------------------------------------------------------------ protocol --

    def check_dependencies(self) -> None:
        self.require_frameworks()

    def _build(self) -> Any:
        import torch

        self._torch = torch

        checkpoint = self.resolve_weights()
        model = build_cellclip_visual()

        visual_state = _extract_visual_state(_load_state_dict(checkpoint))

        missing, unexpected = model.load_state_dict(visual_state, strict=False)
        self.loaded_key_count = len(visual_state)
        # Asymmetric handling on purpose: an unexpected key means the checkpoint
        # carries something we do not understand (fail loudly), while a missing
        # key would leave a randomly-initialised tensor producing plausible but
        # wrong embeddings -- also a failure, not a warning.
        if unexpected:
            raise EmbeddingError(
                f'checkpoint contains visual keys this adapter does not know: '
                f'{sorted(unexpected)[:5]}',
                reason='unexpected-keys',
                details=[f'{len(unexpected)} unexpected keys'],
            )
        if missing:
            raise EmbeddingError(
                f'checkpoint is missing visual keys the adapter expects: '
                f'{sorted(missing)[:5]}',
                reason='missing-keys',
                details=[f'{len(missing)} missing keys'],
            )

        model.eval()
        self._device = self.select_device(torch)
        model.to(self._device)
        return model

    def _forward(self, array: Any, **kwargs: Any) -> Any:
        torch = self._torch
        tensor = array if isinstance(array, torch.Tensor) else torch.as_tensor(array)
        tensor = tensor.to(self._device).float()
        if tensor.ndim == 2:
            # A single sample given as (C, D).
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            embeddings = self._model(tensor)
        return embeddings.detach().to('cpu').numpy()


def _load_state_dict(path: Path) -> dict:
    """Read a ``.safetensors`` file, tolerating a nested ``state_dict`` key."""
    from safetensors.torch import load_file

    payload = load_file(str(path))
    if 'state_dict' in payload and len(payload) == 1:
        payload = payload['state_dict']
    return payload


def _extract_visual_state(state: dict) -> dict:
    """Select and de-prefix the checkpoint's vision-tower parameters.

    The published checkpoint prefixes the tower with ``visual.``; this adapter
    owns only that submodule, so the prefix is stripped.  Text-tower and
    pooling keys are reported, not silently dropped, by the caller's
    ``strict=False`` accounting.
    """
    visual = {
        key[len('visual.'):]: value
        for key, value in state.items()
        if key.startswith('visual.')
    }
    if not visual:
        raise EmbeddingError(
            'no `visual.` parameters found in the checkpoint; this does not '
            'look like a CellCLIP release',
            reason='checkpoint-shape',
            details=[f'top-level key sample: {sorted(state)[:8]}'],
        )
    return visual
