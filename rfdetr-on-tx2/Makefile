
DS_HOME ?= /opt/nvidia/deepstream/deepstream/
CUDA_HOME ?= /usr/local/cuda/
TENSORRT_HOME ?=
DEV ?= 0

CXXFLAGS := -Wall -std=c++17 -fPIC -O3

CXXFLAGS +=                     \
  -I$(DS_HOME)/sources/includes \
  -I$(CUDA_HOME)/include

# Add TensorRT include path if TENSORRT_HOME is set, otherwise use local stubs
ifneq ($(TENSORRT_HOME),)
CXXFLAGS += -I$(TENSORRT_HOME)/include
else
CXXFLAGS += -I.
endif

ifeq ($(DEV),1)
CXXFLAGS += -O0 -ggdb3 -Werror
endif

LIBS := /usr/lib/aarch64-linux-gnu/gstreamer-1.0/deepstream/libnvdsgst_infer.so
LDFLAGS := -shared -Wl,-rpath,/usr/lib/aarch64-linux-gnu

TARGET := libdeepstream-rfdetr.so

SRCS := deepstream_rfdetr_bbox.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean lint format version

all: $(TARGET)

$(TARGET): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *.so *~ \#*

lint:
	clang-tidy --extra-arg=-isystem/usr/include/c++/13 --extra-arg=-isystem/usr/include/aarch64-linux-gnu/c++/13/  $(SRCS)

format:
	clang-format --style=file -i $(SRCS)

