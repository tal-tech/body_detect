#include "caffe/layers/Normalize/normalize_layer_base.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseNormalizeLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    this->CheckWeights();
    
    const std::vector<int> input_shape = this->inputs_[0]->shape();
    
    std::vector<int> buffer_shape = { 1, input_shape[1],  input_shape[2], input_shape[3] };
    buffer_.Reshape(buffer_shape);
    
    if (across_spatial_){
      std::vector<int> norm_shape = { input_shape[0], 1, 1, 1 };
      norm_.Reshape(norm_shape);
    }else{
      std::vector<int> norm_shape = { input_shape[0], 1, input_shape[2], input_shape[3] };
      norm_.Reshape(norm_shape);
    }

    
    int channels = input_shape[1]; 
    sum_channel_multiplier_.Reshape(  std::vector<int> { 1, channels, 1, 1 } );
    caffe_cpu_set<Dtype>(channels, Dtype(1), sum_channel_multiplier_.mutable_cpu_data());

    int spatial_dim = input_shape[2] * input_shape[3];
    sum_spatial_multiplier_.Reshape(  std::vector<int> { 1, 1, input_shape[2], input_shape[3] } );
    caffe_cpu_set<Dtype>(spatial_dim, Dtype(1), sum_spatial_multiplier_.mutable_cpu_data());
    
    this->outputs_[0]->Reshape(input_shape);
  }

  INSTANTIATE_CLASS(BaseNormalizeLayer);

} // namespace facethink
