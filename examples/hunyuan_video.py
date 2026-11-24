"""Connect Sol³ to HunyuanVideo's sparse joint-attention call."""

from sol3_integrations import hunyuan_video


def generate(pipeline, target, *args, release_points=(), fusion_modules=(), **kwargs):
    """Pass (owner, attribute) for the joint-attention sparse branch."""
    # The host packs video first and passes the valid text suffix and its
    # protection settings through the original call.
    with hunyuan_video.activate(target=target, release_points=release_points, fusion_modules=fusion_modules):
        return pipeline(*args, **kwargs)
