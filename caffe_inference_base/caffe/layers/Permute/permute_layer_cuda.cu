#include "caffe/layers/Permute/permute_layer_cuda.hpp"

namespace facethink {

  template <typename Dtype>
  __global__ void PermuteKernel(const int nthreads,
				const Dtype* const bottom_data, const int* permute_order,
				const int* old_steps, const int* new_steps, const int num_axes,
				Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int temp_idx = index;
      int old_idx = 0;
      for (int i = 0; i < num_axes; ++i) {
	int order = permute_order[i];
	old_idx += (temp_idx / new_steps[i]) * old_steps[order];
	temp_idx %= new_steps[i];
      }

      top_data[index] = bottom_data[old_idx];
    }
  }

  template <typename Dtype>
  void CUDAPermuteLayer<Dtype>::ForwardComputation() {
    if (this->need_permute_){
      const Dtype* input_data = this->inputs_[0]->gpu_data();
      Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
      int num_axes = this->inputs_[0]->shape().size();
      int count = this->inputs_[0]->count();

      const int* permute_order_gpu = this->permute_order_blob_.gpu_data();
      const int* old_steps_gpu = this->old_steps_.gpu_data();
      const int* new_steps_gpu = this->new_steps_.gpu_data();
     
      // NOLINT_NEXT_LINE(whitespace/operators)
      PermuteKernel<Dtype><<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>(
										  count,
										  input_data,
										  permute_order_gpu,
										  old_steps_gpu,
										  new_steps_gpu,
										  num_axes,
										  output_data);
      CUDA_POST_KERNEL_CHECK;
    }
  }

  INSTANTIATE_LAYER_GPU_FORWARD(CUDAPermuteLayer);
} //namespace facethink
