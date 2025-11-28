

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


  gst-launch-1.0 -e \
  filesrc location=input.mp4 ! decodebin ! nvvideoconvert ! \
  "video/x-raw(memory:NVMM),format=NV12" ! queue ! mux.sink_0 \
  nvstreammux name=mux width=2048 height=1366 batch-size=1 ! \
  nvinfer config-file-path=rtfdet_bbox_config.txt ! \
  queue ! nvdsosd ! \
  nvvidconv ! "video/x-raw(memory:NVMM),format=NV12" ! \
  nvv4l2h264enc ! h264parse ! mp4mux ! \
  filesink location=output.mp4



  gst-launch-1.0 -e \
  filesrc location=input.mp4 ! decodebin ! \
  videoconvert ! video/x-raw,format=RGBA ! \
  nvvidconv ! "video/x-raw(memory:NVMM),format=NV12" ! \
  queue ! mux.sink_0 \
  nvstreammux name=mux width=2048 height=1366 batch-size=1 ! \
  nvinfer config-file-path=rtfdet_bbox_config.txt ! \
  queue ! nvdsosd ! \
  nvvidconv ! "video/x-raw(memory:NVMM),format=NV12" ! \
  nvv4l2h264enc ! h264parse ! mp4mux ! \
  filesink location=output.mp4


  gst-launch-1.0 -e \
  filesrc location=input.mp4 ! qtdemux name=demux \
  demux.video_0 ! h264parse ! nvv4l2decoder ! \
  "video/x-raw(memory:NVMM),format=NV12" ! \
  nvstreammux name=mux width=2048 height=1366 batch-size=1 ! \
  nvinfer config-file-path=rtfdet_bbox_config.txt ! \
  nvdsosd ! \
  nvv4l2h264enc ! h264parse ! mp4mux ! \
  filesink location=output.mp4


gst-launch-1.0 -e \
 filesrc location=input.mp4 ! qtdemux name=demux \
 demux.video_0 ! h264parse ! nvv4l2decoder ! mux.sink_0 \
 nvstreammux name=mux width=2048 height=1376 batch-size=1 live-source=0 \
 ! nvinfer config-file-path=rtfdet_bbox_config.txt ! \
 nvdsosd ! \
 nvvideoconvert ! "video/x-raw(memory:NVMM),format=I420" ! \
 nvv4l2h264enc ! h264parse ! mp4mux ! \
 filesink location=output.mp4



#doesn't crash, no output
gst-launch-1.0 -e filesrc location=input.mp4 ! qtdemux name=demux demux.video_0 ! queue ! h264parse ! nvv4l2decoder ! nvvideoconvert ! "video/x-raw(memory:NVMM),width=2048,height=1368" ! mux.sink_0 nvstreammux name=mux width=2048 height=1368 batch-size=1 live-source=0 ! nvinfer config-file-path=rtfdet_bbox_config.txt ! nvdsosd ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=NV12" ! nvv4l2h264enc ! h264parse ! mp4mux ! filesink location=output.mp4