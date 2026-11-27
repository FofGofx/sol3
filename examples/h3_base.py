"""Call Sol³ from H3-Base's sparse joint-attention branch on SM80."""

from sol3_attn import get_preset
from sol3_integrations import h3_base


_H3_CONFIG = get_preset("h3")["kernel_config"]


def joint_attention(q, k, v, *, cfg=None, prefix_tokens, dense):
    """Use the H3-Base preset or supply a config from the host controller."""
    # Pack conditioning tokens before video. Request dense output for prefix
    # queries as well as exact prefix keys/values for the video queries.
    return h3_base.attention(
        q, k, v,
        cfg=_H3_CONFIG if cfg is None else dict(cfg, compute_prefix=True),
        sink_start=0,
        sink_tokens=prefix_tokens,
        dense=dense,
    )
