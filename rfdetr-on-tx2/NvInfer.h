/* Stub header for TensorRT NvInfer.h
 * This provides minimal forward declarations for nvinfer1 namespace types
 * that are referenced by nvdsinfer_custom_impl.h but not actually used
 * in this codebase.
 *
 * On systems with TensorRT development headers installed, set TENSORRT_HOME
 * in the Makefile to use the real headers instead.
 */

#ifndef NV_INFER_H
#define NV_INFER_H

namespace nvinfer1
{
    class INetworkDefinition;
    class IBuilder;
    class IBuilderConfig;
    class ICudaEngine;
    class IPluginFactory;
    
    enum class DataType : int
    {
        kFLOAT = 0,
        kHALF = 1,
        kINT8 = 2,
        kINT32 = 3,
        kBOOL = 4
    };
}

#endif // NV_INFER_H

