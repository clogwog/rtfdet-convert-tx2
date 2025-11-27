import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

engine_file = "rfdetr_nano.engine"
input_image = "input.jpg"
output_image = "output.jpg"

# ----------------------------
# Load TensorRT engine
# ----------------------------
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(engine_file)
context = engine.create_execution_context()

# ----------------------------
# Prepare input
# ----------------------------
img = cv2.imread(input_image)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
img_input = np.transpose(img_resized, (2, 0, 1))[None, :, :, :]  # 1x3x224x224

# Allocate device memory
d_input = cuda.mem_alloc(img_input.nbytes)
# Adjust output size according to your ONNX export (example below)
output_size = 300 * 91  # dummy, adjust to number of dets * num_classes
d_output = cuda.mem_alloc(output_size * 4)  # float32

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

# Copy input to device
cuda.memcpy_htod_async(d_input, img_input, stream)

# ----------------------------
# Run inference
# ----------------------------
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
np_output = np.empty(output_size, dtype=np.float32)
cuda.memcpy_dtoh_async(np_output, d_output, stream)

stream.synchronize()

# ----------------------------
# Postprocess (dummy example)
# ----------------------------
# reshape to (num_dets, 91) if that's your exported output
np_output = np_output.reshape(-1, 91)

# Draw dummy boxes for illustration
for i in range(5):  # top 5 detections (replace with actual coords)
    x1, y1, x2, y2 = 20*i, 20*i, 20*i+50, 20*i+50
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"class_{i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imwrite(output_image, img)
print("✅ Output saved:", output_image)

