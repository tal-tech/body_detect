#include "caffe/layers/DepthWiseConvolution/depthwise_conv_layer_builder.hpp"
#include "caffe/util/blob_util.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/DepthWiseConvolution/depthwise_conv_layer_cuda.hpp"
#else
#include "caffe/layers/DepthWiseConvolution/depthwise_conv_layer_cpu.hpp"
#endif


namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  DepthwiseConvLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int num_output;
    bool bias_term;
    int kernel_size_h, kernel_size_w;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    int group;

    this->ParseLayerParameters(layer_param,
                               num_output,
                               kernel_size_h, kernel_size_w,
                               pad_h, pad_w,
                               stride_h, stride_w,
                               dilation_h, dilation_w,
                               bias_term, group, name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDADepthwiseConvLayer<Dtype> >(num_output,
                                                             kernel_size_h, kernel_size_w,
                                                             pad_h, pad_w,
                                                             stride_h, stride_w,
                                                             dilation_h, dilation_w,
                                                             bias_term, group, name);
#else
#endif
    return layer;
  }

} //namespace facethink
