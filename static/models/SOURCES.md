# Models served to the browser (Streamlit static serving: `static/` → `/app/static/`)

## Real-ESRGAN General x4v3 (ONNX, float) — used by `core/superres.py`

- Model: Real-ESRGAN `realesr-general-x4v3` (SRVGGNetCompact, ~1.21 M parameters), 4× super-resolution.
- Export: Qualcomm AI Hub Models, release v0.65.0, ONNX float (QAIRT 2.50, ONNX Runtime 1.27.1).
  - Model card: https://huggingface.co/qualcomm/Real-ESRGAN-General-x4v3
  - Download: https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/real_esrgan_general_x4v3/releases/v0.65.0/real_esrgan_general_x4v3-onnx-float.zip
- Licence: BSD-3-Clause (the model card's licence; it refers to the original implementation's licence,
  https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE). Original work: Xintao Wang et al., Real-ESRGAN.
- Files (unchanged from the zip):

| File | Bytes | SHA-256 |
|---|---|---|
| real_esrgan_general_x4v3.onnx | 161087 | 09e6b675a0a18a97e1057c8dd7e984ef878a0a02ce104430da525b6241c858e7 |
| real_esrgan_general_x4v3.data | 4836096 | 512d0ec9940c2e9d85d27f2952f12a0b77b7841dc22df4ce9f3ea458bc98f37f |

- Graph: IR 10, opset 21; ops Conv, PRelu, DepthToSpace, Resize, Add. The weights are external data
  (`real_esrgan_general_x4v3.data`), passed to ONNX Runtime Web with the `externalData` session option.
- Tensors (from the zip's metadata.json):
  - input `image`: float32, NCHW, fixed shape [1, 3, 128, 128], RGB, values 0..1
  - output `upscaled_image`: float32, NCHW, [1, 3, 512, 512], RGB, values 0..1 (not clamped by the graph)
- Runtime: onnxruntime-web 1.30.0 from jsDelivr (`ort.webgpu.min.js`), WebGPU execution provider only (without
  WebGPU the feature shows a notice and the original imagery).
