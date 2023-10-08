#include "caffe/layers/Convolution/conv_layer_cuda.hpp"
#include "caffe/util/math_func.hpp"
#include "caffe/util/im2col.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDAConvLayer<Dtype>::InitLayer() {
    if (cublas_handle_ == nullptr){
      CHECK_CUBLAS(cublasCreate(&cublas_handle_));
    }

    num_spatial_axes_ = 2;  // here, we only implement the case num_spatial_axes = 2;
    channel_axis_ = 1;
    first_spatial_axis_ = channel_axis_ + 1;
    
    // Special case: im2col is the identity for 1x1 convolution with stride 1
    // and no padding, so flag for skipping the buffer and transformation.
    is_1x1_ = true;
    is_1x1_ &= this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1 &&
      this->stride_h_ == 1 && this->stride_w_ == 1 &&
      this->pad_h_ == 0 && this->pad_w_ == 0;
  }

  template <typename Dtype>
  CUDAConvLayer<Dtype>::~CUDAConvLayer() {
    if (cublas_handle_) {
      CHECK_CUBLAS(cublasDestroy(cublas_handle_));
    }
  }

  template <typename Dtype>
  void CUDAConvLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    this->CheckWeights();

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    std::vector<int> output_shape;
    this->ComputeOutputShape(input_shape, output_shape);
    this->outputs_[0]->Reshape(output_shape);

    conv_in_channels_ = input_shape[channel_axis_];
    conv_in_height_ = input_shape[channel_axis_ + 1];
    conv_in_width_  = input_shape[channel_axis_ + 2];
    
    conv_out_channels_ = this->num_output_;
    conv_out_spatial_dim_ = this->outputs_[0]->count(first_spatial_axis_);
	
  
    kernel_dim_ = this->weights_[0]->count(1);
    weight_offset_ = conv_out_channels_ * kernel_dim_ / this->group_;

    // col bufffer
    std::vector<int> col_buffer_shape;
    col_buffer_shape.push_back(kernel_dim_ * this->group_);
    for (int i = 0; i < num_spatial_axes_; ++i) {
      col_buffer_shape.push_back(output_shape[i+first_spatial_axis_]);
    }
    if (col_buffer_ == nullptr){
      col_buffer_ = std::make_shared<Blob<Dtype> >();
    }
    col_buffer_->Reshape(col_buffer_shape);
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;

    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / this->group_;

    // bias
    out_spatial_dim_ = this->outputs_[0]->count(first_spatial_axis_);
    if (this->has_bias_term()) {
      std::vector<int> bias_multiplier_shape(1, out_spatial_dim_);
      bias_multiplier_.Reshape(bias_multiplier_shape);
      caffe_cpu_set(bias_multiplier_.count(), Dtype(1),
		    bias_multiplier_.mutable_cpu_data());
    }
  }

  template <typename Dtype>
  void CUDAConvLayer<Dtype>::ForwardComputation() {
    this->CheckWeights();

    const Dtype* weight_data = this->weights_[0]->gpu_data();
    const Dtype* input_data = this->inputs_[0]->gpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
    
    int input_dim = this->inputs_[0]->count(channel_axis_);
    int output_dim = this->outputs_[0]->count(channel_axis_);

    const int num = this->inputs_[0]->count(0, 1);  // channel_axis = 1;
    for (int n = 0; n < num; ++n) {

      this->forward_gpu_gemm(input_data + n * input_dim,
       			     weight_data,
       			     output_data + n * output_dim);
      

      if (this->has_bias_term()) {	
	const Dtype* bias_data = this->weights_[1]->gpu_data();
        this->forward_gpu_bias(output_data + n * output_dim, bias_data);
      }
    }
  }

  template <typename Dtype>
  void CUDAConvLayer<Dtype>::forward_gpu_gemm(const Dtype* input, const Dtype* weights,
					      Dtype* output) {
    const Dtype* col_buff = input;
    if (!is_1x1_){
      conv_im2col_gpu(input, col_buffer_->mutable_gpu_data());
      col_buff = col_buffer_->gpu_data();
    }

    for (int g = 0; g < this->group_; ++g){
      caffe_gpu_gemm<Dtype>(cublas_handle_,
			    CblasNoTrans, CblasNoTrans,
			    conv_out_channels_ / this->group_,
			    conv_out_spatial_dim_,
			    kernel_dim_,
			    (Dtype)1.,
			    weights + weight_offset_ * g,
			    col_buff + col_offset_ * g,
			    (Dtype)0.,
			    output + output_offset_ * g);
    }
  }

  template <typename Dtype>
  void CUDAConvLayer<Dtype>::conv_im2col_gpu(const Dtype* data, Dtype* col_buff){      
    im2col_gpu(data, conv_in_channels_,
	       conv_in_height_, conv_in_width_,
	       this->kernel_size_h_, this->kernel_size_w_,
	       this->pad_h_, this->pad_w_,
	       this->stride_h_, this->stride_w_,
	       this->dilation_h_, this->dilation_w_,
	       col_buff);
  }

  template <typename Dtype>
  void CUDAConvLayer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias){
    caffe_gpu_gemm<Dtype>(cublas_handle_,
			  CblasNoTrans, CblasNoTrans,
			  this->num_output_,
			  out_spatial_dim_,
			  1,
			  (Dtype)1.,
			  bias,
			  bias_multiplier_.gpu_data(),
			  (Dtype)1.,
			  output);
  }

  INSTANTIATE_CLASS(CUDAConvLayer);
  
} // namespace facethink
