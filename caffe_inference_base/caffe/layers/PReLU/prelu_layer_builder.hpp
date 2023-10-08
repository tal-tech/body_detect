#ifndef __FACETHINK_CAFFE_LAYERS_PRELU_LAYER_BUILDR_HPP__
#define __FACETHINK_CAFFE_LAYERS_PRELU_LAYER_BUILDR_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/layers/PReLU/prelu_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class PReluLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

	void ImportData(const caffe::LayerParameter& layer_param,std::shared_ptr<BaseLayer<Dtype> >& layer);

  private:

	  bool IsShapeEqual(const caffe::LayerParameter& layer_param, std::shared_ptr<BasePReluLayer<Dtype> >& layer);

    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      bool& inplace,
			      bool& channel_shared,
			      std::string& name);

  }; // class PReluLayerBuilder

  REGISTER_LAYER_BUILDER(PReluLayerBuilder, PReLU);

} // namespace facethink

#endif
