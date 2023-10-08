#ifndef __FACETHINK_CAFFE_LAYERS_ABSVAL_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_ABSVAL_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class AbsValLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      std::string& name);
  }; // class AbsValLayerBuilder

  REGISTER_LAYER_BUILDER(AbsValLayerBuilder, AbsVal);

} // namespace facethink


#endif
