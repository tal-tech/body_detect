#include "caffe/layers/ShuffleChannel/shuffle_channel_layer_cuda.hpp"

namespace facethink {
  
  template <typename Dtype>
  __global__ void ShuffleChannelKernel(const int nthreads,
				       const int feature_map_size,
				       Dtype *output, const Dtype *input,
				       int group_row,
				       int group_column,
				       int len) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / group_row / group_column;
      const int i = (index / group_column) % group_row;
      const int j = index % group_column;

      const Dtype* p_i = input + n * feature_map_size + (i * group_column + j) * len;
      Dtype* p_o = output + n * feature_map_size + (j * group_row + i) * len;

      for (int k = 0; k < len; k++)
	p_o[k] = p_i[k];
    }
  }

  template <typename Dtype>
  void CUDAShuffleChannelLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->gpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
    
    int count = this->num_ * this->group_column_ * this->group_row_;
    ShuffleChannelKernel<Dtype> << <CAFFE_GET_BLOCKS(count),
				    CAFFE_CUDA_NUM_THREADS >> >(
								count,
								this->feature_map_size_,
								output_data,
								input_data,
								this->group_row_,
								this->group_column_,
								this->sp_sz_);
    
  }

  INSTANTIATE_CLASS(CUDAShuffleChannelLayer);
}// namespace facethink
