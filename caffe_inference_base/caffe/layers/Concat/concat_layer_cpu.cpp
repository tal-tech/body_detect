#include "caffe/layers/Concat/concat_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {
  
  template <typename Dtype>
  void ConcatLayer<Dtype>::ForwardComputation() {
    if (this->inputs_.size() == 1) { return; }

    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();

    int offset_concat_axis = 0;
    const int output_concat_axis = this->outputs_[0]->shape(this->concat_axis_canonical_);

    for (int i = 0; i < this->inputs_.size(); ++i) {
      const Dtype* input_data = this->inputs_[i]->cpu_data();
      const int input_concat_axis = this->inputs_[i]->shape(this->concat_axis_canonical_);

      const int input_concat_size = input_concat_axis * this->concat_input_size_;
      for (int n = 0; n < this->num_concats_; ++n) {
	caffe_cpu_copy<Dtype>(input_concat_size,
			      input_data + n * input_concat_axis * this->concat_input_size_,
			      output_data + (n * output_concat_axis + offset_concat_axis)
			      * this->concat_input_size_);
      }
      offset_concat_axis += input_concat_axis;
    }
  }

  INSTANTIATE_CLASS(ConcatLayer);
} // namespace facethink
