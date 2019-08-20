#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__

#include <cassert>
#include <iostream>
#include <cudnn.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "mathFunctions.h"

/*
#define CHECK(status)                                                                                           \
    {                                                                                                                           \
        if (status != 0)                                                                                                \
        {                                                                                                                               \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__                                                        \
                      << std::endl;                                                                     \
            abort();                                                                                                    \
        }                                                                                                                          \
    }
*/

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;


/**********************************************************************************/
// Reshape Plugin Layer
/**********************************************************************************/
//SSD Reshape layer : shape{0,-1,2}
template<int OutC>
class Reshape : public IPlugin {
public:
    Reshape() {}

    Reshape(const void *buffer, size_t size) {
        assert(size == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t *>(buffer);
    }

    int getNbOutputs() const override {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
        assert((inputs[0].d[0]) * (inputs[0].d[1]) % OutC == 0);
        //return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);//faster rcnn : shape{2,-1,0}
        return DimsCHW(1, inputs[0].d[0] * inputs[0].d[2] / OutC, OutC);    // shape{0,-1,2}
    }

    int initialize() override {
        return 0;
    }

    void terminate() override {

    }

    size_t getWorkspaceSize(int) const override {
        return 0;
    }

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override {
        CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
        return 0;
    }

    size_t getSerializationSize() override {
        return sizeof(mCopySize);
    }

    void serialize(void *buffer) override {
        *reinterpret_cast<size_t *>(buffer) = mCopySize;
    }

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override {
        mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }

protected:
    size_t mCopySize;
};


/**********************************************************************************/
// Flatten Plugin Layer
/**********************************************************************************/
class FlattenLayer : public IPlugin {
public:
    FlattenLayer() {}

    FlattenLayer(const void *buffer, size_t size) {
        assert(size == 3 * sizeof(int));
        const int *d = reinterpret_cast<const int *>(buffer);
        _size = d[0] * d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return 1; };

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override {
        assert(1 == nbInputDims);
        assert(0 == index);
        assert(3 == inputs[index].nbDims);
        _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        return DimsCHW(_size, 1, 1);
    }

    int initialize() override {
        return 0;
    }

    inline void terminate() override {

    }

    inline size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override {
        CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], batchSize * _size * sizeof(float), cudaMemcpyDeviceToDevice,
                              stream));
        return 0;
    }

    size_t getSerializationSize() override {
        return 3 * sizeof(int);
    }

    void serialize(void *buffer) override {
        int *d = reinterpret_cast<int *>(buffer);
        d[0] = dimBottom.c();
        d[1] = dimBottom.h();
        d[2] = dimBottom.w();
    }

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};


/**********************************************************************************/
// Softmax Plugin Layer
/**********************************************************************************/
class SoftmaxPlugin : public IPlugin {
public:
    SoftmaxPlugin() {};

    SoftmaxPlugin(const void *buffer, size_t size);

    inline int getNbOutputs() const override { return 1; };

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    int initialize() override;

    inline void terminate() override;

    inline size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override;

protected:
    DimsCHW dimsBottomData;
    int mInputC, mInputH, mInputW;
    float *scale_data;
    int count, outer_num_, inner_num_, channels;
};


/**********************************************************************************/
// Concat Plugin Layer: N×(C*H*W), C*H*W不同，(C*H*W)上的连接），输入数为3，适用于mbox_conf和mbox_loc和mbox_priorbox层
/**********************************************************************************/
class ConcatPlugin : public IPlugin {
public:
    ConcatPlugin(int axis) { _axis = axis; };

    ConcatPlugin(int axis, const void *buffer, size_t size);

    inline int getNbOutputs() const override { return 1; };

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    int initialize() override;

    inline void terminate() override;

    inline size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override;

protected:
    DimsCHW dimsIncep, dimsConv3, dimsConv4;
    int inputs_size;
    int top_concat_axis;    //top 层 concat后的维度
    int *bottom_concat_axis = new int[9];   //记录每个bottom层concat维度的shape
    int *concat_input_size_ = new int[9];
    int *num_concats_ = new int[9];
    int _axis;
};


/**********************************************************************************/
// Concatenation Plugin Layer（N×C×H×W，H×W相同，通道数C上的连接）：输入数量为2，适用于conv1_concat和conv2_concat层
/**********************************************************************************/
class ConcatenationPlugin : public IPlugin {
public:
    ConcatenationPlugin(int axis) { _axis_two = axis; };

    ConcatenationPlugin(int axis, const void *buffer, size_t size);

    inline int getNbOutputs() const override { return 1; };

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    int initialize() override;

    inline void terminate() override;

    inline size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override;

protected:
    DimsCHW dimsA, dimsB;
    int inputs_size_two;
    int top_concat_axis_two;    //top 层 concat后的维度
    int *bottom_concat_axis_two = new int[9];//记录每个bottom层concat维度的shape
    int *concat_input_size_two = new int[9];
    int *num_concats_two = new int[9];
    int _axis_two;
};


/**********************************************************************************/
// Concatenation Plugin Layer（N×C×H×W，H×W相同，通道数C上的连接）：输入数量为4，适用于inception_a1_concat和inception_a2_concat和inception_a3_concat层
/**********************************************************************************/
class ConcatenationFourPlugin : public IPlugin {
public:
    ConcatenationFourPlugin(int axis) { _axis_four = axis; };

    ConcatenationFourPlugin(int axis, const void *buffer, size_t size);

    inline int getNbOutputs() const override { return 1; };

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    int initialize() override;

    inline void terminate() override;

    inline size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void *buffer) override;

    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override;

protected:
    DimsCHW dimsA1, dimsA2, dimsA3, dimsA4;
    int inputs_size_four;
    int top_concat_axis_four;    //top 层 concat后的维度
    int *bottom_concat_axis_four = new int[18];//记录每个bottom层concat维度的shape
    int *concat_input_size_four = new int[18];
    int *num_concats_four = new int[18];
    int _axis_four;
};


/**********************************************************************************/
// PluginFactory
/**********************************************************************************/
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory {
public:
    virtual nvinfer1::IPlugin *
    createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights) override;

    IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override;

    void (*nvPluginDeleter)(INvPlugin *) { [](INvPlugin *ptr) { ptr->destroy(); }};

    bool isPlugin(const char *name) override;

    void destroyPlugin();

    // permute layer
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> inception_a3_concat_mbox_loc_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> inception_a3_concat_mbox_conf_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv3_2_mbox_loc_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv3_2_mbox_conf_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv4_2_mbox_loc_perm_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv4_2_mbox_conf_perm_layer{nullptr, nvPluginDeleter};

    // priorbox layer
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> inception_a3_concat_mbox_priorbox_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv3_2_mbox_priorbox_layer{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv4_2_mbox_priorbox_layer{nullptr, nvPluginDeleter};

   // concatenation + concat layer
    std::unique_ptr<ConcatenationPlugin> conv1_concat_layer{nullptr};               // ConcatenationPlugin (C通道上,2输入)
    std::unique_ptr<ConcatenationPlugin> conv2_concat_layer{nullptr};
    std::unique_ptr<ConcatenationFourPlugin> inception_a1_concat_layer{nullptr};    // ConcatenationFourPlugin (C通道上,4输入)
    std::unique_ptr<ConcatenationFourPlugin> inception_a2_concat_layer{nullptr};
    std::unique_ptr<ConcatenationFourPlugin> inception_a3_concat_layer{nullptr};
    std::unique_ptr<ConcatPlugin> mbox_loc_layer{nullptr};                          // ConcatPlugin （C*H*W）上
    std::unique_ptr<ConcatPlugin> mbox_conf_layer{nullptr};
    std::unique_ptr<ConcatPlugin> mbox_priorbox_layer{nullptr};

    // reshape layer
    std::unique_ptr<Reshape<2>> mbox_conf_reshape_layer{nullptr};

    // flatten layer
    std::unique_ptr<FlattenLayer> inception_a3_concat_mbox_loc_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> inception_a3_concat_mbox_conf_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> conv3_2_mbox_loc_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> conv3_2_mbox_conf_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> conv4_2_mbox_loc_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> conv4_2_mbox_conf_flat_layer{nullptr};
    std::unique_ptr<FlattenLayer> mbox_conf_flatten_layer{nullptr};

    // softmax layer
    std::unique_ptr<SoftmaxPlugin> mbox_conf_softmax_layer{nullptr};

    // detection output layer
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> detection_out_layer{nullptr, nvPluginDeleter};
};

#endif
