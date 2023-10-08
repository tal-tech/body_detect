#include "caffe/layers/Permute/permute_layer_cpu.hpp"

namespace facethink {

  template <typename Dtype>
  void Permute(const int count, Dtype* input_data, const bool forward,
	       const int* permute_order, const int* old_steps, const int* new_steps,
	       const int num_axes, Dtype* output_data) {
    for (int i = 0; i < count; ++i) {
      int old_idx = 0;
      int idx = i;
      for (int j = 0; j < num_axes; ++j) {
        int order = permute_order[j];
        old_idx += (idx / new_steps[j]) * old_steps[order];
        idx %= new_steps[j];
      }
      if (forward) {
        output_data[i] = input_data[old_idx];
      } else {
        input_data[old_idx] = output_data[i];
      }
    }
  }
  
  template <typename Dtype>
  void PermuteLayer<Dtype>::ForwardComputation() {
    if (this->need_permute_) {
      Dtype* input_data = this->inputs_[0]->mutable_cpu_data();
      Dtype* output_data = this->outputs_[0]->mutable_cpu_data();

      int num_axes = this->inputs_[0]->shape().size();
      const int count = this->inputs_[0]->count();
      
      const int* permute_order = this->permute_order_blob_.cpu_data();
      const int* old_steps = this->old_steps_.cpu_data();
      const int* new_steps = this->new_steps_.cpu_data();
      
      bool forward = true;
      Permute(count, input_data, forward, permute_order, old_steps,
	      new_steps, num_axes, output_data);
    } 
  }

  INSTANTIATE_CLASS(PermuteLayer);
}
