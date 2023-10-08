#include "caffe/layers/ROIPooling/roi_pooling_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseROIPoolingLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    int channels = this->inputs_[0]->shape(1);
    int height   = this->inputs_[0]->shape(2);
    int width    = this->inputs_[0]->shape(3);
    int num = this->inputs_[1]->shape(0);

    std::vector<int> output_shape{num, channels, pooled_h_, pooled_w_};
    this->outputs_[0]->Reshape(output_shape);

    max_idx_.Reshape(output_shape);
  }

  INSTANTIATE_CLASS(BaseROIPoolingLayer);

} // namespace facethink
