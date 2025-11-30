no luck.. spend 2 days trying to change the model to something that the TX2 doesn't complain about
only to get a runtime error during the optimisation
```
Starting program: /usr/bin/gst-launch-1.0 -e filesrc location=input.mp4 \! qtdemux name=demux demux.video_0 \! queue \! h264parse \! nvv4l2decoder \! nvvideoconvert \! video/x-raw\(memory:NVMM\),width=2048,height=1368 \! mux.sink_0 nvstreammux name=mux width=2048 height=1368 batch-size=1 live-source=0 \! nvinfer config-file-path=rfdetr.txt \! nvdsosd \! nvvideoconvert \! video/x-raw\(memory:NVMM\),format=NV12 \! nvv4l2h264enc \! h264parse \! mp4mux \! filesink location=output.mp4
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1".
[New Thread 0x7fb6ad5170 (LWP 22756)]
[New Thread 0x7fb62d4170 (LWP 22757)]
[New Thread 0x7fb5ad3170 (LWP 22758)]
[New Thread 0x7fb52d2170 (LWP 22759)]
[New Thread 0x7faa4d9170 (LWP 22760)]
Setting pipeline to PAUSED ...
Opening in BLOCKING MODE
[New Thread 0x7f83d99170 (LWP 22761)]
Opening in BLOCKING MODE
0:00:01.998921093 22753   0x5555c63f50 INFO                 nvinfer gstnvinfer.cpp:638:gst_nvinfer_logger:<nvinfer0> NvDsInferContext[UID 1]: Info from NvDsInferContextImpl::buildModel() <nvdsinfer_context_impl.cpp:1914> [UID = 1]: Trying to create engine from model files
WARNING: [TRT]: onnx2trt_utils.cpp:366: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
WARNING: [TRT]: [RemoveDeadLayers] Input Tensor input is unused or used only at compile-time, but is not being removed.
[New Thread 0x7f6c529170 (LWP 22762)]
gst-launch-1.0: /root/gpgpu/MachineLearning/myelin/src/compiler/optimizer/const_ppg.cpp:1184: void myelin::ir::unop_fold(myelin::ir::operation_t*, size_t, size_t, size_t, size_t, const symbolic_shape_t&, const symbolic_shape_t&, const symbolic_shape_t&, output_type*, output_type*) [with output_type = float; size_t = long unsigned int; myelin::symbolic_shape_t = std::vector<myelin::symbolic_value_t>]: Assertion `0' failed.

Thread 1 "gst-launch-1.0" received signal SIGABRT, Aborted.
__GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:51
51	../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
(gdb) bt
#0  __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:51
#1  0x0000007fb7b84974 in __GI_abort () at abort.c:79
#2  0x0000007fb7b7cd3c in __assert_fail_base (fmt=0x7fb7c77cf0 "%s%s%s:%u: %s%sAssertion `%s' failed.\n%n", assertion=assertion@entry=0x7f882ea328 "0", file=file@entry=0x7f882c7488 "/root/gpgpu/MachineLearning/myelin/src/compiler/optimizer/const_ppg.cpp", line=line@entry=1184,
    function=function@entry=0x7f882c6390 "void myelin::ir::unop_fold(myelin::ir::operation_t*, size_t, size_t, size_t, size_t, const symbolic_shape_t&, const symbolic_shape_t&, const symbolic_shape_t&, output_type*, output_type*) [with output"...) at assert.c:92
#3  0x0000007fb7b7cdbc in __GI___assert_fail (assertion=0x7f882ea328 "0", file=0x7f882c7488 "/root/gpgpu/MachineLearning/myelin/src/compiler/optimizer/const_ppg.cpp", line=1184,
    function=0x7f882c6390 "void myelin::ir::unop_fold(myelin::ir::operation_t*, size_t, size_t, size_t, size_t, const symbolic_shape_t&, const symbolic_shape_t&, const symbolic_shape_t&, output_type*, output_type*) [with output"...) at assert.c:101
#4  0x0000007f87324624 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#5  0x0000007f87323f5c in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#6  0x0000007f87314680 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#7  0x0000007f87308a80 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#8  0x0000007f8730e674 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#9  0x0000007f8756cad8 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#10 0x0000007f8757566c in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#11 0x0000007f87577008 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#12 0x0000007f874eb2ec in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#13 0x0000007f8751543c in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#14 0x0000007f864b7bfc in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#15 0x0000007f8649a1e0 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#16 0x0000007f8649ad0c in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#17 0x0000007f8651dfd0 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#18 0x0000007f867defb8 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#19 0x0000007f867dfc8c in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#20 0x0000007f8651cf0c in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#21 0x0000007f8654a1f8 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#22 0x0000007f867c63c4 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#23 0x0000007f866a871c in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#24 0x0000007f8668d7b0 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#25 0x0000007f866a5888 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#26 0x0000007f86544714 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#27 0x0000007f865489d0 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#28 0x0000007f8680b460 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#29 0x0000007f8680ff3c in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#30 0x0000007fa9ae9504 in ?? () from /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so
#31 0x0000007fa9ae95fc in ?? () from /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so
#32 0x0000007fa9aecb94 in ?? () from /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so
#33 0x0000007fa9ace55c in ?? () from /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so
#34 0x0000007fa9acebe8 in ?? () from /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so
#35 0x0000007fa9ad0580 in ?? () from /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so
#36 0x0000007fa9ad0eb0 in createNvDsInferContext(INvDsInferContext**, _NvDsInferContextInitParams&, void*, void (*)(INvDsInferContext*, unsigned int, NvDsInferLogLevel, char const*, void*)) () from /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so
#37 0x0000007fa9b968b0 in ?? () from /usr/lib/aarch64-linux-gnu/gstreamer-1.0/deepstream/libnvdsgst_infer.so
#38 0x0000007fb792b224 in ?? () from /usr/lib/aarch64-linux-gnu/libgstbase-1.0.so.0
#39 0x0000005555b7edb0 in ?? ()
Backtrace stopped: previous frame inner to this frame (corrupt stack?)
```


The issue is not with the model preprocessing (which works ) but with TensorRT's internal optimizer failing on complex transformer operations. This is a TensorRT bug, not a model issue.
