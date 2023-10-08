#include "caffe/layers/Eltwise/eltwise_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void EltwiseLayer<Dtype>::ForwardComputation() {
    const int count = this->outputs_[0]->count();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();

    int* mask = nullptr;
    const Dtype* input_data = nullptr;
    
    switch (this->op_) {
    case EltwiseOp::PROD:
      caffe_cpu_mul(count, this->inputs_[0]->cpu_data(), this->inputs_[1]->cpu_data(), output_data);
      for (size_t i = 2; i < this->inputs_.size(); ++i) {
	caffe_cpu_mul(count, output_data, this->inputs_[i]->cpu_data(), output_data);
      }
      break;
      
    case EltwiseOp::SUM:
      caffe_cpu_set(count, Dtype(0), output_data);
      // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
      if (this->coeffs_.empty()) 
	this->coeffs_.assign(this->inputs_.size(), Dtype(1.0));
    
      for (size_t i = 0; i < this->inputs_.size(); ++i) {
	caffe_cpu_axpy(count, this->coeffs_[i], this->inputs_[i]->cpu_data(), output_data);
      }
      break;
      
    case EltwiseOp::MAX:
      // Initialize
      mask = this->max_idx_.mutable_cpu_data();
      caffe_cpu_set(count,  0, mask);
      caffe_cpu_copy(count, this->inputs_[0]->cpu_data(), output_data);
  
      // bottom 2++
      for (size_t blob_idx = 1; blob_idx < this->inputs_.size(); ++blob_idx) {
	input_data = this->inputs_[blob_idx]->cpu_data();
	for (int idx = 0; idx < count; ++idx) {
	  if (input_data[idx] > output_data[idx]) {
	    output_data[idx] = input_data[idx];  // maxval
	    mask[idx] = blob_idx;  // maxid
	  }
	}
      }
      break;
      
    default:
      BOOST_LOG_TRIVIAL(error)<<"EltwiseLayer: Unknown elementwise operation.";
    }

  }

  INSTANTIATE_CLASS(EltwiseLayer);
}
