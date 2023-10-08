#include "caffe/layers/Concat/concat_layer_cuda.hpp"

namespace facethink {

  template <typename Dtype>
  __global__ void Concat(const int nthreads, const Dtype* in_data,
			 const bool forward, const int num_concats, const int concat_size,
			 const int output_concat_axis, const int input_concat_axis,
			 const int offset_concat_axis, Dtype* out_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int total_concat_size = concat_size * input_concat_axis;
      const int concat_num = index / total_concat_size;
      const int concat_index = index % total_concat_size;
      const int out_index = concat_index +
        (concat_num * output_concat_axis + offset_concat_axis) * concat_size;
      if (forward) {
	out_data[out_index] = in_data[index];
      } else {
	out_data[index] = in_data[out_index];
      }
    }
  }
  
  template <typename Dtype>
  void CUDAConcatLayer<Dtype>::ForwardComputation() {
    if (this->inputs_.size() == 1) { return; }
    
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
    
    int offset_concat_axis = 0;
    const int output_concat_axis = this->outputs_[0]->shape(this->concat_axis_canonical_);
    
    for (int i = 0; i < this->inputs_.size(); ++i) {
      const Dtype* input_data = this->inputs_[i]->gpu_data();
      const int input_concat_axis = this->inputs_[i]->shape(this->concat_axis_canonical_);
      
      const int input_concat_size = input_concat_axis * this->concat_input_size_;
      const int nthreads = input_concat_size * this->num_concats_;
      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
								 nthreads,
								 input_data,
								 true,
								 this->num_concats_,
								 this->concat_input_size_,
								 output_concat_axis,
								 input_concat_axis,
								 offset_concat_axis,
								 output_data);
      CUDA_POST_KERNEL_CHECK;
      offset_concat_axis += input_concat_axis;
    }
  }

  INSTANTIATE_LAYER_GPU_FORWARD(CUDAConcatLayer);
}
