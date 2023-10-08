#ifndef __FACETHINK_CAFFE_LAYERS_DEPTHWISE_CONV_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_DEPTHWISE_CONV_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/layers/Convolution/conv_layer_base.hpp"
#include "caffe/layers/Convolution/conv_layer_builder.hpp"

namespace facethink {

  template <typename Dtype>
  class DepthwiseConvLayerBuilder: public ConvLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

  private:

  }; // class DepthWiseConvLayerBuilder

  REGISTER_LAYER_BUILDER(DepthwiseConvLayerBuilder, DepthwiseConvolution);
} // namespace facethink

#endif
