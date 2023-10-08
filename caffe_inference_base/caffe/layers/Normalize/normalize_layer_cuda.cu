#include "caffe/layers/Normalize/normalize_layer_cuda.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {
    
  // divid a matrix with vector
  template <typename Dtype>
  __global__ void DivBsx(const int nthreads, const Dtype* A,
			 const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
			 Dtype* B) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int c = index % cols;
      int r = (index / cols) % rows;
      if (trans == CblasNoTrans) {
	B[index] = A[index] / v[c];
      } else {
	B[index] = A[index] / v[r];
      }
    }
  }

  template <typename Dtype>
  __global__ void MulBsx(const int nthreads, const Dtype* A,
			 const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
			 Dtype* B) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int c = index % cols;
      int r = (index / cols) % rows;
      if (trans == CblasNoTrans) {
	B[index] = A[index] * v[c];
      } else {
	B[index] = A[index] * v[r];
      }
    }
  }

  template <typename Dtype>
  void CUDANormalizeLayer<Dtype>::ForwardComputation(){  
    const std::vector<int> input_shape = this->inputs_[0]->shape();
    const Dtype* input_data = this->inputs_[0]->gpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
   
    Dtype* norm_data;
    if (this->across_spatial_) {
      // need to index it
      norm_data = this->norm_.mutable_cpu_data();
    } else {
      norm_data = this->norm_.mutable_gpu_data();
      // add eps to avoid overflow
      caffe_gpu_set<Dtype>(this->norm_.count(), Dtype(this->eps_), norm_data);
    }
    
    const Dtype* sum_channel_multiplier = this->sum_channel_multiplier_.gpu_data();
    int num = input_shape[0];
    int dim = input_shape[1] * input_shape[2] * input_shape[3];
    int spatial_dim = input_shape[2] * input_shape[3];
    int channels = input_shape[1];
    
    Dtype* buffer_data = this->buffer_.mutable_gpu_data();
    for (int n = 0; n < num; ++n) {
      caffe_gpu_powx<Dtype>(dim, input_data, Dtype(2), buffer_data);

      if (this->across_spatial_) {
	Dtype normsqr;
	caffe_gpu_asum<Dtype>(cublas_handle_, dim, buffer_data, &normsqr);
	// add eps to avoid overflow
	norm_data[n] = pow(normsqr + this->eps_, Dtype(0.5));
	Dtype alpha =  Dtype(1.0 / norm_data[n]);
	caffe_gpu_scale<Dtype>(cublas_handle_, dim, alpha, input_data, output_data);
      }else{
	// compute norm
	caffe_gpu_gemv<Dtype>(cublas_handle_,
			      CblasTrans, channels, spatial_dim, Dtype(1),
			      buffer_data, sum_channel_multiplier, Dtype(1),
			      norm_data);
	
	caffe_gpu_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
	// scale the layer
	// NOLINT_NEXT_LINE(whitespace/operators)
        DivBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
									  dim,
									  input_data,
									  norm_data,
									  channels,
									  spatial_dim,
									  CblasNoTrans,
									  output_data);
	CUDA_POST_KERNEL_CHECK;
	norm_data += spatial_dim;
      }

      const Dtype* scale_data;
      if (this->channel_shared_) {
	scale_data = this->weights_[0]->cpu_data();
      } else {
	scale_data = this->weights_[0]->gpu_data();
      }
      // scale the output
      if (this->channel_shared_) {
        caffe_gpu_scal<Dtype>(cublas_handle_, dim, scale_data[0], output_data);
      } else {
	// NOLINT_NEXT_LINE(whitespace/operators)
	MulBsx<Dtype><<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
									 dim,
									 output_data,
									 scale_data,
									 channels,
									 spatial_dim,
									 CblasTrans,
									 output_data);
	CUDA_POST_KERNEL_CHECK;
      }
      input_data  += dim;
      output_data += dim;
    }
  }

  INSTANTIATE_LAYER_GPU_FORWARD(CUDANormalizeLayer);
} // namespace facethink
