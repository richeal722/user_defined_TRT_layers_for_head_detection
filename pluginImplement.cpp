#include "pluginImplement.h"
#include <vector>
#include <algorithm>
//#include "mathFunctions.h"


void trt_copy(const int N, const float *X, float *Y) {
    if (X != Y) {
        CUDA_CHECK(cudaMemcpy(Y, X, sizeof(float) * N, cudaMemcpyDefault));
    }
}


/**********************************************************************************/
// Softmax Plugin Layer
/**********************************************************************************/


/**********************************************************************************/
// Concat Plugin Layer
/**********************************************************************************/
ConcatPlugin::ConcatPlugin(int axis, const void *buffer, size_t size) {
    assert(size == (9 * sizeof(int)));
    const int *d = reinterpret_cast<const int *>(buffer);

    dimsIncep = DimsCHW{d[0], d[1], d[2]};
    dimsConv3 = DimsCHW{d[3], d[4], d[5]};
    dimsConv4 = DimsCHW{d[6], d[7], d[8]};

    _axis = axis;
}

Dims ConcatPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) {
    assert(nbInputDims == 3);

    if (_axis == 1) {
        top_concat_axis = inputs[0].d[0] + inputs[1].d[0] + inputs[2].d[0];
        return DimsCHW(top_concat_axis, 1, 1);
    } else if (_axis == 2) {
        top_concat_axis = inputs[0].d[1] + inputs[1].d[1] + inputs[2].d[1];
        return DimsCHW(2, top_concat_axis, 1);
    } else {
        return DimsCHW(0, 0, 0);
    }
}

int ConcatPlugin::initialize() {
    inputs_size = 3;    // 3个bottom�?
    if (_axis == 1)     // c
    {
        top_concat_axis = dimsIncep.c() + dimsConv3.c() + dimsConv4.c();
        bottom_concat_axis[0] = dimsIncep.c();
        bottom_concat_axis[1] = dimsConv3.c();
        bottom_concat_axis[2] = dimsConv4.c();


        concat_input_size_[0] = dimsIncep.h() * dimsIncep.w();
        concat_input_size_[1] = dimsConv3.h() * dimsConv3.w();
        concat_input_size_[2] = dimsConv4.h() * dimsConv4.w();


        num_concats_[0] = dimsIncep.c();
        num_concats_[1] = dimsConv3.c();
        num_concats_[2] = dimsConv4.c();
    }
    else if (_axis == 2)    // h
    {
        top_concat_axis = dimsIncep.h() + dimsConv3.h() + dimsConv4.h();
        bottom_concat_axis[0] = dimsIncep.h();
        bottom_concat_axis[1] = dimsConv3.h();
        bottom_concat_axis[2] = dimsConv4.h();

        concat_input_size_[0] = dimsIncep.w();
        concat_input_size_[1] = dimsConv3.w();
        concat_input_size_[2] = dimsConv4.w();

        num_concats_[0] = dimsIncep.c() * dimsIncep.h();
        num_concats_[1] = dimsConv3.c() * dimsConv3.h();
        num_concats_[2] = dimsConv4.c() * dimsConv4.h();

    }
    else  //_param.concat_axis == 3 , w
    {
        top_concat_axis = dimsIncep.w() + dimsConv3.w() + dimsConv4.w();
        bottom_concat_axis[0] = dimsIncep.w();
        bottom_concat_axis[1] = dimsConv3.w();
        bottom_concat_axis[2] = dimsConv4.w();


        concat_input_size_[0] = 1;
        concat_input_size_[1] = 1;
        concat_input_size_[2] = 1;

        return 0;
    }

    return 0;
}

void ConcatPlugin::terminate() {
    //CUDA_CHECK(cudaFree(scale_data));
    delete[] bottom_concat_axis;
    delete[] concat_input_size_;
    delete[] num_concats_;
}


int ConcatPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) {
    float *top_data = reinterpret_cast<float *>(outputs[0]);
    int offset_concat_axis = 0;
    const bool kForward = true;
    for (int i = 0; i < inputs_size; ++i) {
        const float *bottom_data = reinterpret_cast<const float *>(inputs[i]);

        const int nthreads = num_concats_[i] * concat_input_size_[i];
        //const int nthreads = bottom_concat_size * num_concats_[i];
        ConcatLayer(nthreads, bottom_data, kForward, num_concats_[i], concat_input_size_[i], top_concat_axis,
                    bottom_concat_axis[i], offset_concat_axis, top_data, stream);

        offset_concat_axis += bottom_concat_axis[i];
    }

    return 0;
}

size_t ConcatPlugin::getSerializationSize() {
    return 9 * sizeof(int);
}

void ConcatPlugin::serialize(void *buffer) {
    int *d = reinterpret_cast<int *>(buffer);
    d[0] = dimsIncep.c();
    d[1] = dimsIncep.h();
    d[2] = dimsIncep.w();
    d[3] = dimsConv3.c();
    d[4] = dimsConv3.h();
    d[5] = dimsConv3.w();
    d[6] = dimsConv4.c();
    d[7] = dimsConv4.h();
    d[8] = dimsConv4.w();
}

void ConcatPlugin::configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) {
    dimsIncep = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
    dimsConv3 = DimsCHW{inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]};
    dimsConv4 = DimsCHW{inputs[2].d[0], inputs[2].d[1], inputs[2].d[2]};
}


/**********************************************************************************/
// Concatenation Plugin Layer (2输入)
/**********************************************************************************/
ConcatenationPlugin::ConcatenationPlugin(int axis, const void *buffer, size_t size) {
    assert(size == (6 * sizeof(int)));
    const int *d = reinterpret_cast<const int *>(buffer);

    dimsA = DimsCHW{d[0], d[1], d[2]};
    dimsB = DimsCHW{d[3], d[4], d[5]};

    _axis_two = axis;
}

Dims ConcatenationPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) {
    assert(nbInputDims == 2);

    if (_axis_two == 1) {
        top_concat_axis_two = inputs[0].d[0] + inputs[1].d[0];
        return DimsCHW(top_concat_axis_two, inputs[0].d[1], inputs[0].d[2]); // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    }
    else {
        return DimsCHW(0, 0, 0);
    }
}

int ConcatenationPlugin::initialize() {
    inputs_size_two = 2;    // 2个bottom�?
    if (_axis_two == 1)     // c
    {
        top_concat_axis_two = dimsA.c() + dimsB.c();
        bottom_concat_axis_two[0] = dimsA.c();
        bottom_concat_axis_two[1] = dimsB.c();

        concat_input_size_two[0] = dimsA.h() * dimsA.w();
        concat_input_size_two[1] = dimsB.h() * dimsB.w();

        num_concats_two[0] = dimsA.c();
        num_concats_two[1] = dimsB.c();
    }
    else
    {
        return 0;
    }

    return 0;
}

void ConcatenationPlugin::terminate() {
    //CUDA_CHECK(cudaFree(scale_data));
    delete[] bottom_concat_axis_two;
    delete[] concat_input_size_two;
    delete[] num_concats_two;
}


int ConcatenationPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) {
    float *top_data = reinterpret_cast<float *>(outputs[0]);
    int offset_concat_axis = 0;
    const bool kForward = true;
    for (int i = 0; i < inputs_size_two; ++i) {
        const float *bottom_data = reinterpret_cast<const float *>(inputs[i]);

        const int nthreads = num_concats_two[i] * concat_input_size_two[i];
        //const int nthreads = bottom_concat_size * num_concats_[i];
        ConcatLayer(nthreads, bottom_data, kForward, num_concats_two[i], concat_input_size_two[i], top_concat_axis_two,
                    bottom_concat_axis_two[i], offset_concat_axis, top_data, stream);

        offset_concat_axis += bottom_concat_axis_two[i];
    }

    return 0;
}

size_t ConcatenationPlugin::getSerializationSize() {
    return 6 * sizeof(int);
}

void ConcatenationPlugin::serialize(void *buffer) {
    int *d = reinterpret_cast<int *>(buffer);
    d[0] = dimsA.c();
    d[1] = dimsA.h();
    d[2] = dimsA.w();
    d[3] = dimsB.c();
    d[4] = dimsB.h();
    d[5] = dimsB.w();

}

void ConcatenationPlugin::configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) {
    dimsA = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
    dimsB = DimsCHW{inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]};
}


/**********************************************************************************/
// ConcatenationFour Plugin Layer (4输入)
/**********************************************************************************/
ConcatenationFourPlugin::ConcatenationFourPlugin(int axis, const void *buffer, size_t size) {
    assert(size == (12 * sizeof(int)));
    const int *d = reinterpret_cast<const int *>(buffer);

    dimsA1 = DimsCHW{d[0], d[1], d[2]};
    dimsA2 = DimsCHW{d[3], d[4], d[5]};
    dimsA3 = DimsCHW{d[6], d[7], d[8]};
    dimsA4 = DimsCHW{d[9], d[10], d[11]};

    _axis_four = axis;
}

Dims ConcatenationFourPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) {
    assert(nbInputDims == 4);

    if (_axis_four == 1) {
        top_concat_axis_four = inputs[0].d[0] + inputs[1].d[0] + inputs[2].d[0] + inputs[3].d[0] ;
        return DimsCHW(top_concat_axis_four, inputs[0].d[1], inputs[0].d[2]); // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    }
    else {
        return DimsCHW(0, 0, 0);
    }
}

int ConcatenationFourPlugin::initialize() {
    inputs_size_four = 4;    // 2个bottom�?
    if (_axis_four == 1)     // c
    {
        top_concat_axis_four = dimsA1.c() + dimsA2.c() + dimsA3.c() + dimsA4.c();
        bottom_concat_axis_four[0] = dimsA1.c();
        bottom_concat_axis_four[1] = dimsA2.c();
        bottom_concat_axis_four[2] = dimsA3.c();
        bottom_concat_axis_four[3] = dimsA4.c();

        concat_input_size_four[0] = dimsA1.h() * dimsA1.w();
        concat_input_size_four[1] = dimsA2.h() * dimsA2.w();
        concat_input_size_four[2] = dimsA3.h() * dimsA3.w();
        concat_input_size_four[3] = dimsA4.h() * dimsA4.w();

        num_concats_four[0] = dimsA1.c();
        num_concats_four[2] = dimsA2.c();
        num_concats_four[3] = dimsA3.c();
        num_concats_four[4] = dimsA4.c();
    }
    else
    {
        return 0;
    }

    return 0;
}

void ConcatenationFourPlugin::terminate() {
    //CUDA_CHECK(cudaFree(scale_data));
    delete[] bottom_concat_axis_four;
    delete[] concat_input_size_four;
    delete[] num_concats_four;
}


int ConcatenationFourPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) {
    float *top_data = reinterpret_cast<float *>(outputs[0]);
    int offset_concat_axis = 0;
    const bool kForward = true;
    for (int i = 0; i < inputs_size_four; ++i) {
        const float *bottom_data = reinterpret_cast<const float *>(inputs[i]);

        const int nthreads = num_concats_four[i] * concat_input_size_four[i];
        //const int nthreads = bottom_concat_size * num_concats_[i];
        ConcatLayer(nthreads, bottom_data, kForward, num_concats_four[i], concat_input_size_four[i], top_concat_axis_four,
                    bottom_concat_axis_four[i], offset_concat_axis, top_data, stream);

        offset_concat_axis += bottom_concat_axis_four[i];
    }

    return 0;
}

size_t ConcatenationFourPlugin::getSerializationSize() {
    return 12 * sizeof(int);
}

void ConcatenationFourPlugin::serialize(void *buffer) {
    int *d = reinterpret_cast<int *>(buffer);
    d[0] = dimsA1.c();
    d[1] = dimsA1.h();
    d[2] = dimsA1.w();
    d[3] = dimsA2.c();
    d[4] = dimsA2.h();
    d[5] = dimsA2.w();
    d[6] = dimsA3.c();
    d[7] = dimsA3.h();
    d[8] = dimsA3.w();
    d[9] = dimsA4.c();
    d[10] = dimsA4.h();
    d[11] = dimsA4.w();
}

void ConcatenationFourPlugin::configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) {
    dimsA1 = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
    dimsA2 = DimsCHW{inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]};
    dimsA3 = DimsCHW{inputs[2].d[0], inputs[2].d[1], inputs[2].d[2]};
    dimsA4 = DimsCHW{inputs[3].d[0], inputs[3].d[1], inputs[3].d[2]};
}



/**********************************************************************************/
// PluginFactory
/**********************************************************************************/
nvinfer1::IPlugin *PluginFactory::createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights) {

    assert(isPlugin(layerName));

    // permute layer
    if (!strcmp(layerName, "inception_a3_concat_mbox_loc_perm")) {
        assert(inception_a3_concat_mbox_loc_perm_layer.get() == nullptr);
        inception_a3_concat_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return inception_a3_concat_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "inception_a3_concat_mbox_conf_perm")) {
        assert(inception_a3_concat_mbox_conf_perm_layer.get() == nullptr);
        inception_a3_concat_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return inception_a3_concat_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_loc_perm")) {
        assert(conv3_2_mbox_loc_perm_layer.get() == nullptr);
        conv3_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv3_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_conf_perm")) {
        assert(conv3_2_mbox_conf_perm_layer.get() == nullptr);
        conv3_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv3_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_loc_perm")) {
        assert(conv4_2_mbox_loc_perm_layer.get() == nullptr);
        conv4_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv4_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_conf_perm")) {
        assert(conv4_2_mbox_conf_perm_layer.get() == nullptr);
        conv4_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv4_2_mbox_conf_perm_layer.get();
    }

    // priorbox layer
    else if (!strcmp(layerName, "inception_a3_concat_mbox_priorbox")) {
        assert(inception_a3_concat_mbox_priorbox_layer.get() == nullptr);
        float min_size[3] = {32,64,128}, max_size[0] = {}, aspect_ratio[1] = {1.0};
        inception_a3_concat_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(
                        {min_size, max_size, aspect_ratio, 3, 0, 1, false, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 32.0,
                         32.0, 0.5}), nvPluginDeleter);
        return inception_a3_concat_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_priorbox")) {
        assert(conv3_2_mbox_priorbox_layer.get() == nullptr);
        float min_size[1] = {256}, max_size[0] = {}, aspect_ratio[1] = {1.0};
        conv3_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(
                {min_size, max_size, aspect_ratio, 1, 0, 1, false, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 64.0,
                 64.0, 0.5}), nvPluginDeleter);
        return conv3_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_priorbox")) {
        assert(conv4_2_mbox_priorbox_layer.get() == nullptr);
        float min_size[1] = {512.0}, max_size[0] = {}, aspect_ratio[1] = {1.0};
        conv4_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDPriorBoxPlugin(
                {min_size, max_size, aspect_ratio, 1, 0, 1, false, false, {0.1, 0.1, 0.2, 0.2}, 0, 0, 128.0,
                 128.0, 0.5}), nvPluginDeleter);
        return conv4_2_mbox_priorbox_layer.get();
    }

    // concatenation + concatenationFour + concat layer
    else if (!strcmp(layerName, "conv1_concat")) {                                                          // concatenation
        assert(conv1_concat_layer.get() == nullptr);
        conv1_concat_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1));
        return conv1_concat_layer.get();
    }
    else if (!strcmp(layerName, "conv2_concat")) {
        assert(conv2_concat_layer.get() == nullptr);
        conv2_concat_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1));
        return conv2_concat_layer.get();
    }
    else if (!strcmp(layerName, "inception_a1_concat")) {                                                   // concatenationFour
        assert(inception_a1_concat_layer.get() == nullptr);
        inception_a1_concat_layer = std::unique_ptr<ConcatenationFourPlugin>(new ConcatenationFourPlugin(1));
        return inception_a1_concat_layer.get();
    }
    else if (!strcmp(layerName, "inception_a2_concat")) {
        assert(inception_a2_concat_layer.get() == nullptr);
        inception_a2_concat_layer = std::unique_ptr<ConcatenationFourPlugin>(new ConcatenationFourPlugin(1));
        return inception_a2_concat_layer.get();
    }
    else if (!strcmp(layerName, "inception_a3_concat")) {
        assert(inception_a3_concat_layer.get() == nullptr);
        inception_a3_concat_layer = std::unique_ptr<ConcatenationFourPlugin>(new ConcatenationFourPlugin(1));
        return inception_a3_concat_layer.get();
    }
    else if (!strcmp(layerName, "mbox_loc")) {                                                              // concat
        assert(mbox_loc_layer.get() == nullptr);
        mbox_loc_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1));
        return mbox_loc_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf")) {
        assert(mbox_conf_layer.get() == nullptr);
        mbox_conf_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1));
        return mbox_conf_layer.get();
    }
    else if (!strcmp(layerName, "mbox_priorbox")) {
        assert(mbox_priorbox_layer.get() == nullptr);
        mbox_priorbox_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(2));
        return mbox_priorbox_layer.get();
    }

    // reshape layer
    else if (!strcmp(layerName, "mbox_conf_reshape")) {
        assert(mbox_conf_reshape_layer.get() == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        mbox_conf_reshape_layer = std::unique_ptr<Reshape<2>>(new Reshape<2>());
        return mbox_conf_reshape_layer.get();
    }

    // flatten layer
    else if (!strcmp(layerName, "inception_a3_concat_mbox_loc_flat")) {
        assert(inception_a3_concat_mbox_loc_flat_layer.get() == nullptr);
        inception_a3_concat_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return inception_a3_concat_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "inception_a3_concat_mbox_conf_flat")) {
        assert(inception_a3_concat_mbox_conf_flat_layer.get() == nullptr);
        inception_a3_concat_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return inception_a3_concat_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_loc_flat")) {
        assert(conv3_2_mbox_loc_flat_layer.get() == nullptr);
        conv3_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return conv3_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_conf_flat")) {
        assert(conv3_2_mbox_conf_flat_layer.get() == nullptr);
        conv3_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return conv3_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_loc_flat")) {
        assert(conv4_2_mbox_loc_flat_layer.get() == nullptr);
        conv4_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return conv4_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_conf_flat")) {
        assert(conv4_2_mbox_conf_flat_layer.get() == nullptr);
        conv4_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return conv4_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf_flatten")) {
        assert(mbox_conf_flatten_layer.get() == nullptr);
        mbox_conf_flatten_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mbox_conf_flatten_layer.get();
    }

    // softmax layer
    else if (!strcmp(layerName, "mbox_conf_softmax")) {
        assert(mbox_conf_softmax_layer == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        mbox_conf_softmax_layer = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin());
        return mbox_conf_softmax_layer.get();
    }

    // detection_out layer
    else if (!strcmp(layerName, "detection_out")) {
        assert(detection_out_layer.get() == nullptr);
        detection_out_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDDetectionOutputPlugin({true, false, 0, 2, 500, 150, 0.2, 0.2, CodeType_t::CENTER_SIZE}),
                 nvPluginDeleter);
        return detection_out_layer.get();
    }

    // others
    else {
        std::cout << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

IPlugin *PluginFactory::createPlugin(const char *layerName, const void *serialData, size_t serialLength) {

    assert(isPlugin(layerName));

    // permute layer
    if (!strcmp(layerName, "inception_a3_concat_mbox_loc_perm")) {
        assert(inception_a3_concat_mbox_loc_perm_layer.get() == nullptr);
        inception_a3_concat_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return inception_a3_concat_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "inception_a3_concat_mbox_conf_perm")) {
        assert(inception_a3_concat_mbox_conf_perm_layer.get() == nullptr);
        inception_a3_concat_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return inception_a3_concat_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_loc_perm")) {
        assert(conv3_2_mbox_loc_perm_layer.get() == nullptr);
        conv3_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return conv3_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_conf_perm")) {
        assert(conv3_2_mbox_conf_perm_layer.get() == nullptr);
        conv3_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return conv3_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_loc_perm")) {
        assert(conv4_2_mbox_loc_perm_layer.get() == nullptr);
        conv4_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return conv4_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_conf_perm")) {
        assert(conv4_2_mbox_conf_perm_layer.get() == nullptr);
        conv4_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPermutePlugin(serialData, serialLength), nvPluginDeleter);
        return conv4_2_mbox_conf_perm_layer.get();
    }

    // priorbox layer
    else if (!strcmp(layerName, "inception_a3_concat_mbox_priorbox")) {
        assert(inception_a3_concat_mbox_priorbox_layer.get() == nullptr);
        inception_a3_concat_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return inception_a3_concat_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_priorbox")) {
        assert(conv3_2_mbox_priorbox_layer.get() == nullptr);
        conv3_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return conv3_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_priorbox")) {
        assert(conv4_2_mbox_priorbox_layer.get() == nullptr);
        conv4_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDPriorBoxPlugin(serialData, serialLength), nvPluginDeleter);
        return conv4_2_mbox_priorbox_layer.get();
    }

    // concatenation + concatenationFour + concat layer
    else if (!strcmp(layerName, "conv1_concat")) {
        assert(conv1_concat_layer.get() == nullptr);
        conv1_concat_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1, serialData, serialLength));
        return conv1_concat_layer.get();
    }
    else if (!strcmp(layerName, "conv2_concat")) {
        assert(conv2_concat_layer.get() == nullptr);
        conv2_concat_layer = std::unique_ptr<ConcatenationPlugin>(new ConcatenationPlugin(1, serialData, serialLength));
        return conv2_concat_layer.get();
    }
    else if (!strcmp(layerName, "inception_a1_concat")) {
        assert(inception_a1_concat_layer.get() == nullptr);
        inception_a1_concat_layer = std::unique_ptr<ConcatenationFourPlugin>(new ConcatenationFourPlugin(1, serialData, serialLength));
        return inception_a1_concat_layer.get();
    }
    else if (!strcmp(layerName, "inception_a2_concat")) {
        assert(inception_a2_concat_layer.get() == nullptr);
        inception_a2_concat_layer = std::unique_ptr<ConcatenationFourPlugin>(new ConcatenationFourPlugin(1, serialData, serialLength));
        return inception_a2_concat_layer.get();
    }
    else if (!strcmp(layerName, "inception_a3_concat")) {
        assert(inception_a3_concat_layer.get() == nullptr);
        inception_a3_concat_layer = std::unique_ptr<ConcatenationFourPlugin>(new ConcatenationFourPlugin(1, serialData, serialLength));
        return inception_a3_concat_layer.get();
    }
    else if (!strcmp(layerName, "mbox_loc")) {
        assert(mbox_loc_layer.get() == nullptr);
        mbox_loc_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1, serialData, serialLength));
        return mbox_loc_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf")) {
        assert(mbox_conf_layer.get() == nullptr);
        mbox_conf_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1, serialData, serialLength));
        return mbox_conf_layer.get();
    }
    else if (!strcmp(layerName, "mbox_priorbox")) {
        assert(mbox_priorbox_layer.get() == nullptr);
        mbox_priorbox_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(2, serialData, serialLength));
        return mbox_priorbox_layer.get();
    }

    // reshape layer
    else if (!strcmp(layerName, "mbox_conf_reshape")) {
        assert(mbox_conf_reshape_layer == nullptr);
        mbox_conf_reshape_layer = std::unique_ptr<Reshape<2>>(new Reshape<2>(serialData, serialLength));
        return mbox_conf_reshape_layer.get();
    }

    // flatten layer
    else if (!strcmp(layerName, "inception_a3_concat_mbox_loc_flat")) {
        assert(inception_a3_concat_mbox_loc_flat_layer.get() == nullptr);
        inception_a3_concat_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return inception_a3_concat_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "inception_a3_concat_mbox_conf_flat")) {
        assert(inception_a3_concat_mbox_conf_flat_layer.get() == nullptr);
        inception_a3_concat_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return inception_a3_concat_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_loc_flat")) {
        assert(conv3_2_mbox_loc_flat_layer.get() == nullptr);
        conv3_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return conv3_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv3_2_mbox_conf_flat")) {
        assert(conv3_2_mbox_conf_flat_layer.get() == nullptr);
        conv3_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return conv3_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_loc_flat")) {
        assert(conv4_2_mbox_loc_flat_layer.get() == nullptr);
        conv4_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return conv4_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv4_2_mbox_conf_flat")) {
        assert(conv4_2_mbox_conf_flat_layer.get() == nullptr);
        conv4_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return conv4_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf_flatten")) {
        assert(mbox_conf_flatten_layer.get() == nullptr);
        mbox_conf_flatten_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData, serialLength));
        return mbox_conf_flatten_layer.get();
    }

    // softmax layer
    else if (!strcmp(layerName, "mbox_conf_softmax")) {
        assert(mbox_conf_softmax_layer == nullptr);
        mbox_conf_softmax_layer = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin(serialData, serialLength));
        return mbox_conf_softmax_layer.get();
    }

    // detection_out layer
    else if (!strcmp(layerName, "detection_out")) {
        assert(detection_out_layer.get() == nullptr);
        detection_out_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(
                createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
        return detection_out_layer.get();
    }

    // others
    else {
        assert(0);
        return nullptr;
    }
}

bool PluginFactory::isPlugin(const char *name) {
    return (
               !strcmp(name, "inception_a3_concat_mbox_loc_perm")   // permute layer
            || !strcmp(name, "inception_a3_concat_mbox_conf_perm")
            || !strcmp(name, "conv3_2_mbox_loc_perm")
            || !strcmp(name, "conv3_2_mbox_conf_perm")
            || !strcmp(name, "conv4_2_mbox_loc_perm")
            || !strcmp(name, "conv4_2_mbox_conf_perm")
            || !strcmp(name, "inception_a3_concat_mbox_priorbox")   // priorbox layer
            || !strcmp(name, "conv3_2_mbox_priorbox")
            || !strcmp(name, "conv4_2_mbox_priorbox")
            || !strcmp(name, "conv1_concat")                        // concatenation layer
            || !strcmp(name, "conv2_concat")
            || !strcmp(name, "inception_a1_concat")
            || !strcmp(name, "inception_a2_concat")
            || !strcmp(name, "inception_a3_concat")
            || !strcmp(name, "mbox_loc")                            // concat layer
            || !strcmp(name, "mbox_conf")
            || !strcmp(name, "mbox_priorbox")
            || !strcmp(name, "mbox_conf_reshape")                   // reshape layer
            || !strcmp(name, "inception_a3_concat_mbox_loc_flat")   // flatten layer
            || !strcmp(name, "inception_a3_concat_mbox_conf_flat")
            || !strcmp(name, "conv3_2_mbox_loc_flat")
            || !strcmp(name, "conv3_2_mbox_conf_flat")
            || !strcmp(name, "conv4_2_mbox_loc_flat")
            || !strcmp(name, "conv4_2_mbox_conf_flat")
            || !strcmp(name, "mbox_conf_flatten")
            || !strcmp(name, "mbox_conf_softmax")                   // softmax layer
            || !strcmp(name, "detection_out"));                     // detection_out layer
}

void PluginFactory::destroyPlugin() {

    // permute layer
    inception_a3_concat_mbox_loc_perm_layer.release();
    inception_a3_concat_mbox_loc_perm_layer = nullptr;
    inception_a3_concat_mbox_conf_perm_layer.release();
    inception_a3_concat_mbox_conf_perm_layer = nullptr;
    conv3_2_mbox_loc_perm_layer.release();
    conv3_2_mbox_loc_perm_layer = nullptr;
    conv3_2_mbox_conf_perm_layer.release();
    conv3_2_mbox_conf_perm_layer = nullptr;
    conv4_2_mbox_loc_perm_layer.release();
    conv4_2_mbox_loc_perm_layer = nullptr;
    conv4_2_mbox_conf_perm_layer.release();
    conv4_2_mbox_conf_perm_layer = nullptr;

    // priorbox layer
    inception_a3_concat_mbox_priorbox_layer.release();
    inception_a3_concat_mbox_priorbox_layer = nullptr;
    conv3_2_mbox_priorbox_layer.release();
    conv3_2_mbox_priorbox_layer = nullptr;
    conv4_2_mbox_priorbox_layer.release();
    conv4_2_mbox_priorbox_layer = nullptr;

    // concatenation + concat layer
    conv1_concat_layer.release();
    conv1_concat_layer = nullptr;
    conv2_concat_layer.release();
    conv2_concat_layer = nullptr;
    inception_a1_concat_layer.release();
    inception_a1_concat_layer = nullptr;
    inception_a2_concat_layer.release();
    inception_a2_concat_layer = nullptr;
    inception_a3_concat_layer.release();
    inception_a3_concat_layer = nullptr;
    mbox_loc_layer.release();
    mbox_loc_layer = nullptr;
    mbox_conf_layer.release();
    mbox_conf_layer = nullptr;
    mbox_priorbox_layer.release();
    mbox_priorbox_layer = nullptr;

    // reshape layer
    mbox_conf_reshape_layer.release();
    mbox_conf_reshape_layer = nullptr;

    // flatten layer
    inception_a3_concat_mbox_loc_flat_layer.release();
    inception_a3_concat_mbox_loc_flat_layer = nullptr;
    inception_a3_concat_mbox_conf_flat_layer.release();
    inception_a3_concat_mbox_conf_flat_layer = nullptr;
    conv3_2_mbox_loc_flat_layer.release();
    conv3_2_mbox_loc_flat_layer = nullptr;
    conv3_2_mbox_conf_flat_layer.release();
    conv3_2_mbox_conf_flat_layer = nullptr;
    conv4_2_mbox_loc_flat_layer.release();
    conv4_2_mbox_loc_flat_layer = nullptr;
    conv4_2_mbox_conf_flat_layer.release();
    conv4_2_mbox_conf_flat_layer = nullptr;
    mbox_conf_flatten_layer.release();
    mbox_conf_flatten_layer = nullptr;

    // softmax layer
    mbox_conf_softmax_layer.release();
    mbox_conf_softmax_layer = nullptr;

    // detection output layer
    detection_out_layer.release();
    detection_out_layer = nullptr;
}
