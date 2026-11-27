# Model integrations

These source modules connect the installed `sol3-attn` wheel to the model's selected sparse call. They ship in the checkout and source archive. To use them from any working directory, set the path once in your launcher, substituting the absolute path to this directory.

```bash
export PYTHONPATH="/path/to/Sol3/integrations${PYTHONPATH:+:$PYTHONPATH}"
```

| Module in `sol3_integrations` | Core profile | Call selected by the host | Example |
| --- | --- | --- | --- |
| `wan22` | `wan`, SM80 | High- and low-noise video self-attention | [wan22.py](../examples/wan22.py) |
| `hunyuan_video` | `hy`, SM80 | Joint attention with the protected text suffix | [hunyuan_video.py](../examples/hunyuan_video.py) |
| `sana_wm` | `sana`, SM80 | Stage 2 Refiner, preserving its native anchor | [sana_wm.py](../examples/sana_wm.py) |
| `ltx25` | `ltx`, SM80 | Stage 1 video self-attention after its capacity probe | [ltx25.py](../examples/ltx25.py) |
| `h3_base` | `h3`, SM80 | Joint attention with protected prefix queries and FFN memory management | [h3_base.py](../examples/h3_base.py) |
| `fasth3` | `fast`, SM120 | Joint attention with text and audio prefixes, returning output and route | [fasth3.py](../examples/fasth3.py) |

The examples plug into an initialized model. For `target`, pass a pair such as `(controller, "sparse_attention")` using the actual owner and attribute in your pipeline. The H3-Base and FastH3 examples show the attention call itself. The LTX example uses a host with `sol3_sparse_binding` and connects the workspace release at the Stage 1 boundary.

### Connecting an existing sparse callable

If the host already supplies a routing configuration, pass it through. When adding a sparse branch to a new host, call `get_preset(profile)` from `sol3_attn` once during setup. Each call returns a fresh copy. Use `preset["kernel_config"]` as the operator configuration. Read `preset["host_policy"]` in the denoiser to select the eligible stage, initial dense steps and protected blocks. Block indices are zero-based. HunyuanVideo uses indices 0 and 20 in the combined double- and single-block sequence.

Compute conditioning positions and lengths from the current token layout.

Pass the module or instance and the name of the sparse callable that the model invokes. For example, in a Wan host whose `controller.sparse_attention(q, k, v, cfg=...)` already receives the model's routing configuration, use the following scope around generation.

```python
from sol3_integrations import wan22

counts = {}
with wan22.activate(target=(controller, "sparse_attention"), counters=counts):
    video = pipeline(prompt)
assert counts.get("sparse_calls", 0) > 0
```

Pass the callable that the model uses for sparse attention. The host controls its dense steps, masks and tensor layout. Exiting the context restores the original callable and releases the workspace. You can also call an adapter directly, for example `h3_base.attention(q, k, v, cfg=cfg, sink_start=0, sink_tokens=prefix, dense=dense)`. The five SM80 adapters accept the host's `cfg` and return the output tensor. FastH3 takes `config`, `prefix_tokens` and `dense`, and returns `(output, route)`.

If your host provides a compatible binding, pass its module as `binding=` or use the default `sol3_sparse_binding` import. Supply the actual host objects that own the release calls and fusion launch sites when entering the scope. A scope with no release points releases its workspace on exit. To preserve earlier release times, register every boundary used by your host, including any function imported by value at the call site.

| Model | Release after these host calls | Fusion module roles |
| --- | --- | --- |
| Wan2.2 | Pass `application=app` for warmup. Register the adapter class's `release` and `close` methods if used. | `pointwise`, `rms_rope` |
| HunyuanVideo | The provider's `release_workspace` | `pointwise`, `rotary` |
| SANA-WM | The refiner adapter class's `__exit__` and the provider's `release_sol3` | `pointwise`, `rotary` |
| LTX-2.5 | The runtime's `release_workspace` and any separate Stage 1 reference used by the pipeline | `pointwise`, `rotary` |
| H3-Base | The runtime's `release_workspace` | `pointwise` |
| FastH3 | The attention provider's `clear_workspace` | `pointwise`, `rotary` |

For example, where the Wan pipeline uses separate pointwise and RMS/RoPE modules:

```python
with wan22.activate(
    target=(controller, "sparse_attention"),
    application=app,
    release_points=[(adapter_class, "release"), (adapter_class, "close")],
    fusion_modules={"pointwise": pointwise_module, "rms_rope": rms_rope_module},
) as scope:
    video = pipeline(prompt)
```

A fusion is installed only when the host kernel's symbol and parsed Triton function body match. For a lazily imported provider, call `scope.attach_fusions(role, module)` after import and before its first launch. For a late-bound release, call `scope.release_after(owner, name)` before the boundary runs. Hooks release workspace after the original operation returns or raises. Exiting the scope restores the original attributes.

The SM80 backends accept `fused_csr` (Wan, HunyuanVideo, SANA-WM and LTX), `parallel_csr` (Wan, HunyuanVideo and LTX), `parallel_csr_chunk` (HunyuanVideo and LTX), and `factorized_risk` (SANA-WM). Existing host spellings are accepted at the input boundary. Where both spellings are supplied, the new key takes precedence before the model's workspace planner applies its tuning.

When implementing a host binding, set `self.__sol3_sparse_model__` on each `BoundSparse` instance to its core profile from the table above, and keep the original callable in `self.owner`. The adapter intercepts calls whose profile matches the active model. Other bindings continue through the host.

H3's FFN guard is in `sol3_integrations.h3_workspace_guard`. Activation can attach to a guard already registered on the transformer. A host integrating the guard for the first time can construct `AttentionScratchGuard(transformer, active_job, release)` at model load, call `reset()` at the start of each job, and `close()` on teardown. `active_job()` returns the current job with `attention_mode="sol_attn^3"` during sparse inference. `release(job)` releases that job's retained attention scratch, including `sol3_attn.release_workspace("h3")` if the core owns it.

For a new model, the wheel's `sol3_attn(q, k, v, *, prefix_tokens=0, dense=None)` works directly without these hooks. The repository adapters use the public `sol3_attn_with_route(q, k, v, *, profile, config, **options)` interface for the existing tuned kernels. Keep model-specific token packing, dense fallback and configuration in the host integration.
