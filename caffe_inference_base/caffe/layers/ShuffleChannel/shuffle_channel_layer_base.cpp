#include "caffe/layers/ShuffleChannel/shuffle_channel_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseShuffleChannelLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    
    this->outputs_[0]->Reshape(this->inputs_[0]->shape());

    num_ = this->inputs_[0]->shape(0);
    feature_map_size_ = this->inputs_[0]->count(1);
    sp_sz_ = this->inputs_[0]->count(2);
    
    const int chs = this->inputs_[0]->shape(1);
    group_row_ = this->group_;
    group_column_ = int(chs / group_row_);
    if (chs != group_column_ * group_row_) {
      BOOST_LOG_TRIVIAL(error)<<"BaseShuffleChannelLayer: Wrong group size";
    }
  }
  
  INSTANTIATE_CLASS(BaseShuffleChannelLayer);
} 
