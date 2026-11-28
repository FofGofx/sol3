"""ComfyUI node "Sol3 Sparse Attention" for MiniMax H3 models (H3-Base, FastH3 and their distilled variants).

Install: copy this folder to ComfyUI/custom_nodes/sol3_attention (ComfyUI 0.46 or later) and install sol3-attn
(`pip install "sol3-attn[sm80]"` on SM80). Put the node between the model loader and the sampler.

The node reuses ComfyUI's built-in "Model Sparse Attention" patch for the sampling window, dense blocks, the
minimum sequence length and H3's packed conditioning rows ([text][cond][ref][audio] before [video]). Only the
kernel differs. Conditioning rows get dense attention and stay exact keys for every video query; video queries go
through the published Sol³ preset for the GPU:
- SM80 (A100, A800): profile "h3", get_preset("h3")["kernel_config"];
- SM120 (RTX PRO 6000 Blackwell): profile "fast", get_preset("fast")["kernel_config"].
The presets' dense-step policy assumes longer schedules than the 2- and 4-step pi-H3 adapters, so the sampling
window stays a node input (start_percent). A call Sol³ cannot take (other GPU, mask, head size other than 128)
runs ComfyUI's normal attention, and the first error of each kind is logged once.
"""
import logging

import torch
import torch.nn.functional as F

import comfy.patcher_extension
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_sparse_attention import BLOCK_SIZE, SparseAttnPatch, parse_block_list

CONFIG = {}  # profile -> kernel_config, loaded on first use so ComfyUI starts without sol3-attn installed


def kernel_config(profile):
    if profile not in CONFIG:
        from sol3_attn import get_preset
        CONFIG[profile] = get_preset(profile)["kernel_config"]
    return CONFIG[profile]


def dense_rows(q, k, v):
    """Dense attention for the conditioning query rows, in Sol³'s [batch, tokens, heads, dim] layout."""
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def sol3_kernel(q, k, v, prefix):
    """One batch item, [1, T, H, 128] BF16; returns the output in the same layout."""
    from sol3_attn import sol3_attn_with_route
    if torch.cuda.get_device_capability(q.device) == (8, 0):
        out, _ = sol3_attn_with_route(q, k, v, profile="h3", config=kernel_config("h3"),
                                      sink_start=0, sink_tokens=prefix, dense=dense_rows)
    else:  # SM120; any other GPU raises here and the caller falls back to dense
        out, _ = sol3_attn_with_route(q, k, v, profile="fast", config=kernel_config("fast"),
                                      prefix_tokens=prefix, dense=dense_rows)
    return out


def make_override(patch, previous):
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        transformer_options = kwargs.get("transformer_options") or {}

        def dense():
            args = (q, k, v, heads)
            kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kwargs)
            return func(*args, **kw) if previous is None else previous(func, *args, **kw)

        if mask is not None or not skip_reshape or q.shape[-1] != 128:
            return dense()
        batch, _, tokens, dim = q.shape                              # ComfyUI passes [batch, heads, tokens, dim]
        reason = patch.dense_reason(transformer_options, tokens, transformer_options.get("block_index"))
        if reason is not None:
            patch.log_once(("dense", tokens, reason), f"dense ({tokens} tokens): {reason}")
            return dense()
        sink, _ = patch.sinks(transformer_options, tokens)
        prefix = min(sink[1] * BLOCK_SIZE, tokens - BLOCK_SIZE)      # a multiple of 64, as the H3 kernel requires
        try:
            out = torch.cat([sol3_kernel(*(t[i:i + 1].transpose(1, 2).to(torch.bfloat16).contiguous()
                                           for t in (q, k, v)), prefix) for i in range(batch)]).to(q.dtype)
        except Exception as error:  # unsupported GPU, missing FlashInfer, out of memory: stay dense
            patch.log_once(("error", type(error).__name__), f"Sol3 failed, running dense: {error!r}"[:600])
            logging.warning("Sol3: %r", error)
            return dense()
        patch.log_once(("sparse", tokens), f"Sol3 sparse: {tokens} tokens, {prefix} prefix rows dense")
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(batch, tokens, heads * dim)

    return override


def install(patch, transformer_options):
    current = transformer_options.get("optimized_attention_override")
    if current in patch.installed:
        return
    override = make_override(patch, current)
    patch.installed.add(override)
    transformer_options["optimized_attention_override"] = override


def cleanup(patch):
    from sol3_attn import release_workspace
    patch.reset()
    release_workspace()


class Sol3SparseAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sol3SparseAttention",
            display_name="Sol3 Sparse Attention (MiniMax H3)",
            category="model/patch",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Sampling progress where sparse attention starts; earlier steps run dense."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
                io.String.Input("dense_blocks", default="0",
                                tooltip="Transformer blocks that always run dense, e.g. 0 or 0,1,47. "
                                        "The paper keeps the first block dense."),
                io.Int.Input("min_tokens", default=12288, min=0, max=1 << 20,
                             tooltip="Shorter sequences run dense."),
                io.Boolean.Input("verbose", default=False),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, start_percent, end_percent, dense_blocks, min_tokens, verbose) -> io.NodeOutput:
        sampling = model.get_model_object("model_sampling")
        patch = SparseAttnPatch(tau=0.0, topk_ratio=0.0, vsa=False,
                                sigma_start=float(sampling.percent_to_sigma(start_percent)),
                                sigma_end=float(sampling.percent_to_sigma(end_percent)),
                                min_tokens=min_tokens, dense_blocks=parse_block_list(dense_blocks),
                                sink_conditioning="exact_kv", extra_tokens=0, verbose=verbose)
        m = model.clone()
        install(patch, m.model_options["transformer_options"])
        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, "sol3",
                                lambda patcher, timestep, options: install(patch, options["transformer_options"]))
        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, "sol3", lambda patcher: cleanup(patch))
        return io.NodeOutput(m)


class Sol3Extension(ComfyExtension):
    async def get_node_list(self):
        return [Sol3SparseAttention]


async def comfy_entrypoint():
    return Sol3Extension()
