#include "caffe/layers/InnerProduct/inner_product_layer_cuda.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDAInnerProductLayer<Dtype>::InitLayer() {
    if (!cublas_handle_) {
      CHECK_CUBLAS(cublasCreate(&cublas_handle_));
    }
  }

  template <typename Dtype>
  CUDAInnerProductLayer<Dtype>::~CUDAInnerProductLayer() {
    if (cublas_handle_){
      CHECK_CUBLAS(cublasDestroy(cublas_handle_));
    }
  }

  template <typename Dtype>
  void CUDAInnerProductLayer<Dtype>::ForwardComputation() {
    this->CheckWeights();

    const Dtype* input_data = this->inputs_[0]->gpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
    
    const Dtype* weight_data = this->weights_[0]->gpu_data();
    
    Dtype alpha = Dtype(1.0);
    Dtype beta  = Dtype(0.0);

    if (this->M_ == 1) {
      caffe_gpu_gemv<Dtype>(cublas_handle_,
			    CblasNoTrans,
			    this->N_, this->K_, (Dtype)1.,
			    weight_data, input_data, (Dtype)0., output_data);
      
      if (this->has_bias_term()) {
	const Dtype* bias_data = this->weights_[1]->gpu_data();
	caffe_gpu_axpy<Dtype>(cublas_handle_,
			      this->N_,
			      this->bias_multiplier_.cpu_data()[0],
			      bias_data,
			      output_data);
      }   
    } else {
      caffe_gpu_gemm<Dtype>(cublas_handle_,
			    CblasNoTrans,
			    this->transpose_ ? CblasNoTrans : CblasTrans,
			    this->M_, this->N_, this->K_,
			    (Dtype)1.,
			    input_data,
			    weight_data,
			    (Dtype)0.,
			    output_data);
      
      if (this->has_bias_term()) {
	const Dtype* bias_data = this->weights_[1]->gpu_data();
	caffe_gpu_gemm<Dtype>(cublas_handle_,
			      CblasNoTrans,
			      CblasNoTrans,
			      this->M_, this->N_, 1,
			      (Dtype)1.,
			      this->bias_multiplier_.gpu_data(),
			      bias_data,
			      (Dtype)1.,
			      output_data);
      }
    }
  }

  INSTANTIATE_CLASS(CUDAInnerProductLayer);
} // namespace facethink
