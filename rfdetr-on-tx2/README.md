

gst-launch-1.0 -e filesrc location=input.mp4 ! decodebin ! queue ! mux.sink_0 \
    nvstreammux name=mux width=1920 height=1080 batch-size=1 ! \
    nvinfer config-file-path=deepstream_rfdetr_bbox_config.txt ! \
    queue ! nvdsosd ! nvv4l2h264enc ! h264parse ! queue ! mp4mux ! \
    filesink location=output.mp4
