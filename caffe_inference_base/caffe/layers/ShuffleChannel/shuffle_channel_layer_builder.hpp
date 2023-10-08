#ifndef __FACETHINK_CAFFE_LAYERS_SHUFFLE_CHANNEL_LAYER_NETBUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_SHUFFLE_CHANNEL_LAYER_NETBUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class ShuffleChannelLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      int& group,
			      std::string& name);
  }; // class ShuffleChannelLayerBuilder

  REGISTER_LAYER_BUILDER(ShuffleChannelLayerBuilder, ShuffleChannel);

} // namespace facethink

#endif
