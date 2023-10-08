#ifndef __FACETHINK_CAFFE_LAYERS_SHUFFLE_CHANNEL_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_SHUFFLE_CHANNEL_LAYER_CPU_HPP__

#include "caffe/layers/ShuffleChannel/shuffle_channel_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class ShuffleChannelLayer: public BaseShuffleChannelLayer<Dtype> {
  public:
    explicit ShuffleChannelLayer(int group = 1,
				 const std::string& name="")
      :BaseShuffleChannelLayer<Dtype>(group, name) {
    }

    virtual void ForwardComputation();

  private:
    void Resize_cpu(Dtype *output, const Dtype *input, int group_row, int group_column, int len);

    DISABLE_COPY_AND_ASSIGN(ShuffleChannelLayer);
  }; // class ShuffleChannelLayer

} // namespace facethink

#endif
