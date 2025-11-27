

gst-launch-1.0 -e filesrc location=input.mp4 ! decodebin ! queue ! mux.sink_0 \
    nvstreammux name=mux width=1920 height=1080 batch-size=1 ! \
    nvinfer config-file-path=deepstream_rfdetr_bbox_config.txt ! \
    queue ! nvdsosd ! nvv4l2h264enc ! h264parse ! queue ! mp4mux ! \
    filesink location=output.mp4



# 2048,1366


gst-launch-1.0 -e filesrc location=input.mp4 ! decodebin ! queue ! mux.sink_0 \
    nvstreammux name=mux width=2048 height=1366 batch-size=1 ! \
    nvinfer config-file-path=rtfdet_bbox_config.txt ! \
    queue ! nvdsosd ! nvv4l2h264enc ! h264parse ! queue ! mp4mux ! \
    filesink location=output.mp4



gst-launch-1.0 -e \
  filesrc location=input.mp4 ! decodebin ! queue ! mux.sink_0 \
  nvstreammux name=mux width=2048 height=1366 batch-size=1 ! \
  nvinfer config-file-path=rtfdet_bbox_config.txt ! \
  queue ! nvdsosd ! \
  "video/x-raw(memory:NVMM),format=NV12" ! \
  nvv4l2h264enc ! h264parse ! mp4mux ! \
  filesink location=output.mp4