#ifndef __FACETHINK_CAFFE_LAYERS_RESHAPE_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_RESHAPE_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class ReshapeLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      std::vector<int>& dims,
			      int& axis,
			      int& num_axes,
			      std::string& name);

  }; // class ReshapeLayerBuilder

  REGISTER_LAYER_BUILDER(ReshapeLayerBuilder, Reshape);

} // namespace facethink


#endif
