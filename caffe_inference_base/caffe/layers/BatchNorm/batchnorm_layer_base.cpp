#include "caffe/layers/BatchNorm/batchnorm_layer_base.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {
  
  template <typename Dtype>
  void BaseBatchNormLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    const int input_count = this->inputs_[0]->count();
    this->outputs_[0]->Reshape(input_shape);
    
    if (input_shape.size() == 1) {
      channels_ = 1;
    } else {
      channels_ = input_shape[1];
    }
   
    std::vector<int> sz = { channels_ };
    mean_.Reshape(sz);
    variance_.Reshape(sz);
    temp_.Reshape(input_shape);
    x_norm_.Reshape(input_shape);
    sz[0] = input_shape[0];
    batch_sum_multiplier_.Reshape(sz);

    int spatial_dim = input_count /(channels_ * input_shape[0]);
    if (spatial_sum_multiplier_.num_axes() == 0 ||
	spatial_sum_multiplier_.shape(0) != spatial_dim) {
      sz[0] = spatial_dim;
      spatial_sum_multiplier_.Reshape(sz);
      Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
      caffe_cpu_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
    }

    int numbychans = channels_ * input_shape[0];
    if (num_by_chans_.num_axes() == 0 ||
	num_by_chans_.shape(0) != numbychans) {
      sz[0] = numbychans;
      num_by_chans_.Reshape(sz);
      caffe_cpu_set(batch_sum_multiplier_.count(), Dtype(1),
		    batch_sum_multiplier_.mutable_cpu_data());
    }
  }

  INSTANTIATE_CLASS(BaseBatchNormLayer);

} // namespace facethink
