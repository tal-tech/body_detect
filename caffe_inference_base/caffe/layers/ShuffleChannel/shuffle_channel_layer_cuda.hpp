#ifndef __FACETHINK_CAFFE_LAYERS_SHUFFLE_CHANNEL_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_SHUFFLE_CHANNEL_LAYER_CUDA_HPP__

#include "caffe/layers/ShuffleChannel/shuffle_channel_layer_base.hpp"


namespace facethink {

  template <typename Dtype>
  class CUDAShuffleChannelLayer: public BaseShuffleChannelLayer<Dtype> {
  public:
    explicit CUDAShuffleChannelLayer(int group = 1,
				     const std::string& name="")
      :BaseShuffleChannelLayer<Dtype>(group, name) {
    }

    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "ShuffleChannel(CUDA)";
    }

    DISABLE_COPY_AND_ASSIGN(CUDAShuffleChannelLayer);
  }; // class CUDAShuffleChannelLayer

} // namespace facethink


#endif
