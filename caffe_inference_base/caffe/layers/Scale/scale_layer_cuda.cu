#include "caffe/layers/Scale/scale_layer_cuda.hpp"

namespace facethink {

  template <typename Dtype>
  __global__ void ScaleForward(const int n, const Dtype* in,
			       const Dtype* scale, const int scale_dim, const int inner_dim,
			       Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      const int scale_index = (index / inner_dim) % scale_dim;
      out[index] = in[index] * scale[scale_index];
    }
  }

  template <typename Dtype>
  __global__ void ScaleBiasForward(const int n, const Dtype* in,
				   const Dtype* scale, const Dtype* bias,
				   const int scale_dim, const int inner_dim, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      const int scale_index = (index / inner_dim) % scale_dim;
      out[index] = in[index] * scale[scale_index] + bias[scale_index];
    }
  }

  
  template <typename Dtype>
  void CUDAScaleLayer<Dtype>::ForwardComputation() {
    
    const Dtype* input_data = this->inputs_[0]->gpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
    const int count = this->inputs_[0]->count();

    std::shared_ptr<Blob<Dtype> > scale = (this->inputs_.size() > 1) ?
      this->inputs_[1] : this->weights_[0];
    const Dtype* scale_data = scale->gpu_data();

    
    if (this->bias_term_) {
      std::shared_ptr<Blob<Dtype> > bias = this->weights_[this->weights_.size() - 1];
      const Dtype* bias_data = bias->gpu_data();

      ScaleBiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
							      count,
							      input_data,
							      scale_data,
							      bias_data,
							      this->scale_dim_,
							      this->inner_dim_,
							      output_data);
    } else {
      ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
							      count,
							      input_data,
							      scale_data,
							      this->scale_dim_,
							      this->inner_dim_,
							      output_data);
    }
    
  }

  INSTANTIATE_CLASS(CUDAScaleLayer);
} // namespace facethink
