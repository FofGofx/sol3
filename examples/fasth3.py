"""Call Sol³ from FastH3's sparse joint-attention branch on SM120."""

from sol3_attn import get_preset
from sol3_integrations import fasth3


_FAST_CONFIG = get_preset("fast")["kernel_config"]


def joint_attention(q, k, v, *, config=None, text_tokens, audio_tokens, dense):
    """Q/K/V contain a text and audio prefix followed by video tokens."""
    # Use the actual packed lengths. The host's dense callback handles prefix
    # queries over the full KV sequence, and the adapter also returns the route.
    output, route = fasth3.attention(
        q, k, v,
        config=_FAST_CONFIG if config is None else config,
        prefix_tokens=text_tokens + audio_tokens,
        dense=dense,
    )
    return output, route
