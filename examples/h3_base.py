"""Call Sol³ from H3-Base's sparse joint-attention branch on SM80."""

from sol3_integrations import h3_base


def joint_attention(q, k, v, *, cfg, prefix_tokens, dense):
    """Supply the host's routing config and dense(q_prefix, k, v) callback."""
    # Pack conditioning tokens before video. Request dense output for prefix
    # queries as well as exact prefix keys/values for the video queries.
    return h3_base.attention(
        q, k, v,
        cfg=dict(cfg, compute_prefix=True),
        sink_start=0,
        sink_tokens=prefix_tokens,
        dense=dense,
    )
