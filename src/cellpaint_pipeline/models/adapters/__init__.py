"""Built-in embedding adapters.

Each module here implements the
:class:`~cellpaint_pipeline.models.embedding.ModelAdapter` protocol for one
method.  Adapters are imported lazily by the registry, so importing this package
never imports a deep-learning framework.

Adapters are grouped by *how* they consume data rather than by paper family,
because that is what the shared interface has to accommodate:

``deepprofiler``
    Keras graph, ``(N, H, W, C)`` raw 16-bit crops, embedding = intermediate
    activation reduced over space.
``ca_mae``
    torch, ``(N, C, 256, 256)`` uint8, normalisation inside the checkpoint.
``cellclip``
    torch, **not pixels** -- ``(N, 5, 1536)`` precomputed channel embeddings.
``o2vae``
    torch + e2cnn, latent mean; O(2)-invariant VAE family.
``unavailable``
    declared methods whose weights cannot be obtained here.
"""
