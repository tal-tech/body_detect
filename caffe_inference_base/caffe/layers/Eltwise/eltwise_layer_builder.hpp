#ifndef __FACETHINK_CAFFE_LAYERS_ELTWISE_LAYER_NETBUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_ELTWISE_LAYER_NETBUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/layers/Eltwise/eltwise_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class EltwiseLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      EltwiseOp& op,
			      std::vector<Dtype>& coeff,
			      std::string& name);
  }; // class EltwiseLayerBuilder

  REGISTER_LAYER_BUILDER(EltwiseLayerBuilder, Eltwise);

} // namespace facethink


#endif
