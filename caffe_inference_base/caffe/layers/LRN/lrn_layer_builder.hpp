#ifndef __FACETHINK_CAFFE_LAYERS_LRN_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_LRN_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class LRNLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
     std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);
  private:
     void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			       int& size,
			       Dtype& alpha,
			       Dtype& beta,
			       Dtype& k,
			       std::string& name);

  }; // class LRNLayerBuilder

  REGISTER_LAYER_BUILDER(LRNLayerBuilder, LRN);

} // namespace facethink

#endif
