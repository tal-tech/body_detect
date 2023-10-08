#ifndef __FACETHINK_CAFFE_LAYERS_CONV_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_CONV_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/layers/Convolution/conv_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class ConvLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

    void ImportData(const caffe::LayerParameter& layer_param,
		    std::shared_ptr<BaseLayer<Dtype> >& layer);

  protected:
    bool IsShapeEqual(const caffe::LayerParameter& layer_param,
		      std::shared_ptr<BaseConvLayer<Dtype> >& layer);

    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      int& num_output,
			      int& kernel_size_h, int& kernel_size_w,
			      int& pad_h, int& pad_w,
			      int& stride_h, int& stride_w,
			      int& dilation_h, int& dilation_w,
			      bool& bias_term,
			      int& group,
			      std::string& name);

  }; // class ConvLayerBuilder

  REGISTER_LAYER_BUILDER(ConvLayerBuilder, Convolution);
} // namespace facethink

#endif
