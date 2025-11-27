
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
engine_path = "rfdetr_nano.engine"

# Load engine
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Allocate buffers
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # Allocate device memory
    device_mem = cuda.mem_alloc(size * dtype().nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append(device_mem)
    else:
        outputs.append(device_mem)

# Dummy input
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
cuda.memcpy_htod(inputs[0], dummy_input)

# Run inference
with engine.create_execution_context() as context:
    context.execute_v2(bindings)

# Fetch outputs
output_np = np.empty([1, 300, 91], dtype=np.float32)  # adjust shape to your model
cuda.memcpy_dtoh(output_np, outputs[0])
print(output_np.shape, output_np)

