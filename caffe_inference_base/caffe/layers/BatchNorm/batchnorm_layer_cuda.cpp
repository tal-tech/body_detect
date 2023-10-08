#include "caffe/layers/BatchNorm/batchnorm_layer_cuda.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  
  template <typename Dtype>
  void CUDABatchNormLayer<Dtype>::InitLayer() {
    if (!cublas_handle_) {
      CHECK_CUBLAS(cublasCreate(&cublas_handle_));
    }
  }

  template <typename Dtype>
  CUDABatchNormLayer<Dtype>::~CUDABatchNormLayer() {
    if (cublas_handle_) {
      CHECK_CUBLAS(cublasDestroy(cublas_handle_));
    }
  }

  template <typename Dtype>
  void CUDABatchNormLayer<Dtype>::ForwardComputation() {
    this->CheckWeights();

    if (!this->use_global_stats_) {
      BOOST_LOG_TRIVIAL(fatal)<< "BatchNormLayer: Must use pre-trained mean and variances in the TEST phase.";
      return;
    }
    
    const Dtype* input_data = this->inputs_[0]->gpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();

    const std::vector<int> input_shape = this->inputs_[0]->shape();
    const int input_count = this->inputs_[0]->count();
    
    int num = input_shape[0];
    int spatial_dim = input_count /(this->channels_ * input_shape[0]);

    if (this->inputs_[0] != this->outputs_[0]) {
      caffe_gpu_copy(input_count, input_data, output_data);
    }


    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->weights_[2]->cpu_data()[0] == 0 ?
      0 : 1 / this->weights_[2]->cpu_data()[0];
      
    caffe_gpu_scale(cublas_handle_,
		    this->variance_.count(),
		    scale_factor,
		    this->weights_[0]->gpu_data(),
		    this->mean_.mutable_gpu_data());
      
    caffe_gpu_scale(cublas_handle_,
		    this->variance_.count(),
		    scale_factor,
		    this->weights_[1]->gpu_data(),
		    this->variance_.mutable_gpu_data());
    

    // subtract mean
    caffe_gpu_gemm<Dtype>(cublas_handle_,
			  CblasNoTrans,
			  CblasNoTrans,
			  num,
			  this->channels_,
			  1,
			  Dtype(1.),
			  this->batch_sum_multiplier_.gpu_data(),
			  this->mean_.gpu_data(),
			  0.,
			  this->num_by_chans_.mutable_gpu_data());
    
    caffe_gpu_gemm<Dtype>(cublas_handle_,
			  CblasNoTrans,
			  CblasNoTrans,
			  this->channels_ * num,
			  spatial_dim,
			  1,
			  Dtype(-1),
			  this->num_by_chans_.gpu_data(),
			  this->spatial_sum_multiplier_.gpu_data(),
			  Dtype(1.),
			  output_data);

    // normalize variance
    caffe_gpu_add_scalar(this->variance_.count(),
			 this->eps_,
			 this->variance_.mutable_gpu_data());
    caffe_gpu_powx(this->variance_.count(),
		   this->variance_.gpu_data(),
		   Dtype(0.5),
		   this->variance_.mutable_gpu_data());

    // replicate variance to input size
    caffe_gpu_gemm<Dtype>(cublas_handle_,
			  CblasNoTrans, CblasNoTrans,
			  num, this->channels_, 1,
			  Dtype(1.),
			  this->batch_sum_multiplier_.gpu_data(),
			  this->variance_.gpu_data(),
			  Dtype(0.),
			  this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(cublas_handle_,
			  CblasNoTrans, CblasNoTrans,
			  this->channels_ * num, spatial_dim, 1,
			  Dtype(1.),
			  this->num_by_chans_.gpu_data(),
			  this->spatial_sum_multiplier_.gpu_data(),
			  Dtype(0.),
			  this->temp_.mutable_gpu_data());
    
    caffe_gpu_div(this->temp_.count(), output_data, this->temp_.gpu_data(), output_data);
  }

  INSTANTIATE_CLASS(CUDABatchNormLayer);
} // namespace facethink
