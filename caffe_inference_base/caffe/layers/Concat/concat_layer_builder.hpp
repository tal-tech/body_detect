#ifndef __FACETHINK_CAFFE_LAYERS_CONCAT_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_CONCAT_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class ConcatLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);
  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      int& concat_axis,
			      std::string& name);

  }; // class ConcatLayerBuilder

  REGISTER_LAYER_BUILDER(ConcatLayerBuilder, Concat);

} // namespace facethink

#endif
