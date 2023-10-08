#include "caffe/layers/Eltwise/eltwise_layer_cuda.hpp"
#include "caffe/util/math_func.hpp"
#include <cfloat>

namespace facethink {

    
  template <typename Dtype>
  void CUDAEltwiseLayer<Dtype>::InitLayer() {
    if (cublas_handle_ == nullptr){
      CHECK_CUBLAS(cublasCreate(&cublas_handle_));
    }
  }

  template <typename Dtype>
  CUDAEltwiseLayer<Dtype>::~CUDAEltwiseLayer() {
    if (cublas_handle_){
      CHECK_CUBLAS(cublasDestroy(cublas_handle_));
    }
  }
  
 
  template <typename Dtype>
  __global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
			     const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
			     int* mask) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      Dtype maxval = -FLT_MAX;
      int maxidx = -1;
      if (bottom_data_a[index] > bottom_data_b[index]) {
	// only update for very first bottom_data blob (blob_idx == 0)
	if (blob_idx == 0) {
	  maxval = bottom_data_a[index];
	  top_data[index] = maxval;
	  maxidx = blob_idx;
	  mask[index] = maxidx;
	}
      } else {
	maxval = bottom_data_b[index];
	top_data[index] = maxval;
	maxidx = blob_idx + 1;
	mask[index] = maxidx;
      }
    }
  }


  template <typename Dtype>
  void CUDAEltwiseLayer<Dtype>::ForwardComputation() {
    const int count = this->outputs_[0]->count();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();

    int* mask = nullptr;
   
    switch (this->op_) {
    case EltwiseOp::PROD:
      caffe_gpu_mul(count, this->inputs_[0]->gpu_data(), this->inputs_[1]->gpu_data(),
		    output_data);
      for (int i = 2; i < this->inputs_.size(); ++i) {
	caffe_gpu_mul(count, output_data, this->inputs_[i]->gpu_data(), output_data);
      }
      break;
   
    case EltwiseOp::SUM:
      caffe_gpu_set(count, Dtype(0.), output_data);
      // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
      if (this->coeffs_.empty()) 
	this->coeffs_.assign(this->inputs_.size(), Dtype(1.0));
      
      for (int i = 0; i < this->inputs_.size(); ++i) {
	caffe_gpu_axpy(cublas_handle_,
		       count, this->coeffs_[i], this->inputs_[i]->gpu_data(), output_data);
      }
      break;

    case EltwiseOp::MAX:
      mask = this->max_idx_.mutable_gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
									      count,
									      this->inputs_[0]->gpu_data(),
									      this->inputs_[1]->gpu_data(),
									      0,
									      output_data,
									      mask);
      for (int i = 2; i < this->inputs_.size(); ++i) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
									       count,
									       output_data,
									       this->inputs_[i]->gpu_data(),
									       i-1,
									       output_data,
									       mask);
      }
      break;
      
    default:
      BOOST_LOG_TRIVIAL(error)<<"EltwiseLayer: Unknown elementwise operation.";
    }
  }

  INSTANTIATE_CLASS(CUDAEltwiseLayer);
}
