# ComfyUI node: Sol³ for MiniMax H3

`Sol3 Sparse Attention (MiniMax H3)` patches a MiniMax H3 model in ComfyUI so that video queries in joint attention
run through Sol³, while H3's packed conditioning rows stay dense and remain exact keys for every video query.

## Requirements

- ComfyUI 0.46 or later (the node reuses the built-in "Model Sparse Attention" patch for the sampling window,
  dense blocks and H3's conditioning layout).
- SM80 (A100, A800): `pip install "sol3-attn[sm80]"`, which pins FlashInfer 0.3.1. With PyTorch 2.14 and CUDA 13,
  FlashInfer 0.3.1 must JIT-compile as C++20; start ComfyUI with
  `NVCC_APPEND_FLAGS="-std=c++20"` and a fresh `FLASHINFER_WORKSPACE_BASE`. FlashInfer 0.7.6 fails in `plan()`.
- SM120 (RTX PRO 6000 Blackwell): `pip install sol3-attn`.

## Install

Copy this folder to `ComfyUI/custom_nodes/sol3_attention` and restart ComfyUI.

## Use

Put the node between the patched model and the guider (after any adapter or sigma shift). It loads the published
preset for the GPU once: `get_preset("h3")["kernel_config"]` on SM80 and `get_preset("fast")["kernel_config"]`
on SM120. Any call Sol³ cannot take (another GPU, a mask, a head size other than 128) runs ComfyUI's normal attention.

| Input | Default | Meaning |
| --- | --- | --- |
| `start_percent` | 0.0 | Sampling progress where sparse attention starts; earlier steps run dense. The presets' dense-step policy assumes longer schedules than 2- and 4-step distilled samplers, so this is left to the user. |
| `end_percent` | 1.0 | Sampling progress where sparse attention stops. |
| `dense_blocks` | `0` | Transformer blocks that always run dense. |
| `min_tokens` | 12288 | Shorter sequences run dense. |
| `verbose` | false | Log the dense or sparse decision once per sequence length. |

The first Sol³ run at each new sequence length compiles kernels (in our runs about 42 s extra on an A100 for the
first shape and 10 s for a new one; 3-4 s on an RTX PRO 6000). Warm up at each shape before timing.

## Example

`example_workflow_api.json` is one single-phase API-format graph: start frame and prompt, then sampling, DWD video
decode, audio decode and an MP4. It uses MiniMax H3 FL2VA (int8 ConvRot) with the pi-H3 2-step DWD adapter from
[ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow); put a `first_frame.png` in ComfyUI's `input` folder.

## Measured

MiniMax H3 FL2VA int8 ConvRot with pi-H3 DWD adapters, 1344 x 768, 24 fps, first frame to video and audio,
ComfyUI 0.46, PyTorch 2.14.1+cu130. Median of three runs after warming every setting at each shot shape; seconds of
sampling (decode was 2-4 s and unchanged).

| GPU | Shot | Dense | Sol³, block 0 dense | Sol³, block 0 sparse |
| --- | --- | ---: | ---: | ---: |
| A100 80 GB (preset `h3`) | 124 frames, 2 steps, 39,933 tokens | 34.1 | 21.6 (-37%) | 22.0 |
| A100 80 GB (preset `h3`) | 90 frames, 4 steps, 29,716 tokens | 41.3 | 27.9 (-32%) | 27.8 |
| RTX PRO 6000 (preset `fast`) | 124 frames, 2 steps | 18.8 | 12.8 (-32%) | 12.6 |
| RTX PRO 6000 (preset `fast`) | 90 frames, 4 steps | 22.7 | 16.8 (-26%) | 16.7 |
