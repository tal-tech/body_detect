#include "caffe/layers/ShuffleChannel/shuffle_channel_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/ShuffleChannel/shuffle_channel_layer_cuda.hpp"
#else
#include "caffe/layers/ShuffleChannel/shuffle_channel_layer_cpu.hpp"
#endif


namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  ShuffleChannelLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int group;
    
    ParseLayerParameters(layer_param,
			 group,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAShuffleChannelLayer<Dtype> >(group, 
							      name);
#else
    layer = std::make_shared<ShuffleChannelLayer<Dtype> >(group,
							  name);
#endif
    return layer;
  }

  template <typename Dtype>
  void ShuffleChannelLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							       int& group,
							       std::string& name) {
    this->ParseLayerName(layer_param, name);
    const caffe::ShuffleChannelParameter& sc_param = layer_param.shuffle_channel_param();

    if (sc_param.has_group()) {
      group = sc_param.group();
    } else {
      group = 1;
    }
  }

} // namespace facethink
