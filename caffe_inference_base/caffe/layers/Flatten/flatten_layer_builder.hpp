#ifndef __FACETHINK_CAFFE_LAYERS_FLATTEN_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_FLATTEN_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class FlattenLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);
  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      int& start_axis,
			      int& end_axis,
			      std::string& name);
  }; // class FlattenLayer
  REGISTER_LAYER_BUILDER(FlattenLayerBuilder, Flatten);
} // namespace facethink


#endif
