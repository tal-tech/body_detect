#include "caffe/layers/Scale/scale_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void ScaleLayer<Dtype>::ForwardComputation() {
  
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    const int count = this->inputs_[0]->count();

    std::shared_ptr<Blob<Dtype> > scale = (this->inputs_.size() > 1) ?
      this->inputs_[1] : this->weights_[0];
    const Dtype* scale_data = scale->cpu_data();

   
    if (this->bias_term_) {
      std::shared_ptr<Blob<Dtype> > bias = this->weights_[this->weights_.size() - 1];
      const Dtype* bias_data = bias->cpu_data();

      for (int i = 0; i < count; ++i) {
	const int scale_index = (i / this->inner_dim_) % this->scale_dim_;
	output_data[i] = input_data[i] * scale_data[scale_index] + bias_data[scale_index];
      }
    } else {    
      for (int i = 0; i < count; ++i) {
	const int scale_index = (i / this->inner_dim_) % this->scale_dim_;
	output_data[i] = input_data[i] * scale_data[scale_index];
      }
    }
  }

  INSTANTIATE_CLASS(ScaleLayer);
} // namespace facethink
