"""Call Sol³ from FastH3's sparse joint-attention branch on SM120."""

from sol3_integrations import fasth3


def joint_attention(q, k, v, *, config, text_tokens, audio_tokens, dense):
    """Q/K/V contain a text and audio prefix followed by video tokens."""
    # Use the actual packed lengths. The host's dense callback handles prefix
    # queries over the full KV sequence, and the adapter also returns the route.
    output, route = fasth3.attention(
        q, k, v,
        config=config,
        prefix_tokens=text_tokens + audio_tokens,
        dense=dense,
    )
    return output, route
