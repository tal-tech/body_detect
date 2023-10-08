#include "caffe/layers/AbsVal/absval_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void AbsValLayer<Dtype>::ForwardComputation() {
    const int input_count = this->inputs_[0]->count();
    const Dtype * input_data = this->inputs_[0]->cpu_data();
    Dtype * output_data = this->outputs_[0]->mutable_cpu_data();
    
    caffe_cpu_abs(input_count, input_data, output_data);
  }

  INSTANTIATE_CLASS(AbsValLayer);
}
