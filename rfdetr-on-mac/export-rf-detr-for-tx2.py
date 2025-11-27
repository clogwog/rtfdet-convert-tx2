import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from rfdetr import RFDETRNano
import torch.nn.functional as F

import os
os.makedirs("output", exist_ok=True)


original_interpolate = F.interpolate

def patched_interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False, **kwargs):
    # force antialias=False for bicubic
    if mode == "bicubic":
        antialias = False
    return original_interpolate(input, size=size, scale_factor=scale_factor, mode=mode,
                                align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
                                antialias=antialias, **kwargs)

F.interpolate = patched_interpolate

original_grid_sample = F.grid_sample

def patched_grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    # Replace bicubic GridSample with bilinear interpolation
    # Only works for regular upsample grids (approximation)
    # If your grids are complex attention patterns, this may reduce accuracy
    N, C, H, W = input.shape
    out_H = grid.shape[1] if len(grid.shape) >= 4 else H
    out_W = grid.shape[2] if len(grid.shape) >= 4 else W
    return F.interpolate(input, size=(out_H, out_W), mode='bilinear', align_corners=align_corners)

F.grid_sample = patched_grid_sample



# 1️⃣ Load your RF-DETR Nano model
model = RFDETRNano()

# Force model to CPU for export
model.model.model = model.model.model.to("cpu")

# 2️⃣ Replace LayerNorm with Identity (TRT cannot parse it)
def replace_layernorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, nn.Identity())
        else:
            replace_layernorm(child)

replace_layernorm(model.model.model)
model.model.model.eval()

# 3️⃣ Dummy input on CPU
dummy_input = torch.randn(1, 3, 224, 224)


# 4️⃣ Export to ONNX
onnx_path = "output/inference_model_no_ln.onnx"
torch.onnx.export(
    model.model.model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["dets", "labels"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "dets": {0: "batch"}, "labels": {0: "batch"}},
)

# 5️⃣ Simplify ONNX
model_onnx = onnx.load(onnx_path)
model_simp, check = simplify(model_onnx)
assert check, "Simplified ONNX failed check!"
onnx.save(model_simp, "output/inference_model_no_ln.sim.onnx")
print("✅ Simplified ONNX ready:", "output/inference_model_no_ln.sim.onnx")

# 6️⃣ TensorRT (run on Jetson GPU)
# /usr/src/tensorrt/bin/trtexec --onnx=output/inference_model_no_ln.sim.onnx \
#     --saveEngine=output/rfdetr_nano.engine --fp16 --workspace=4096 --verbose

