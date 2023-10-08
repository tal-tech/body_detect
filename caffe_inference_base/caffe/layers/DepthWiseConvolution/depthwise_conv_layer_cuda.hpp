#ifndef __FACETHINK_CAFFE_LAYERS_DEPTHWISE_CONV_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_DEPTHWISE_CONV_LAYER_CUDA_HPP__

#include "caffe/layers/Convolution/conv_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDADepthwiseConvLayer: public BaseConvLayer<Dtype> {
  public:
    explicit CUDADepthwiseConvLayer(int num_output,
                                    int kernel_size_h, int kernel_size_w,
                                    int pad_h = 0, int pad_w = 0,
                                    int stride_h = 1, int stride_w = 1,
                                    int dilation_h = 1, int dilation_w = 1,
                                    bool bias_term = true,
                                    int group = 1,
                                    const std::string& name = "")
      :BaseConvLayer<Dtype>(num_output,
                            kernel_size_h, kernel_size_w,
                            pad_h, pad_w,
                            stride_h, stride_w,
                            dilation_h, dilation_w,
                            bias_term, group, name) {
      this->InitLayer();
    }

    virtual void InitLayer() {};
    virtual void ForwardShape();
    virtual void ForwardComputation();

    DISABLE_COPY_AND_ASSIGN(CUDADepthwiseConvLayer);
  }; // class CUDADepthwiseConvLayer

} //namespace facethink

#endif
