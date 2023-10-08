#include "caffe/layers/Convolution/conv_layer_cudnn.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDNNConvLayer<Dtype>::InitLayer() {
    if (!cudnn_handle_){
      CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
    
      CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
      if (this->has_bias_term()){
	CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc_));
      }
      CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
      
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc_));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc_));
    }
  }

  template <typename Dtype>
  CUDNNConvLayer<Dtype>::~CUDNNConvLayer() {
    if (cudnn_handle_) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc_));
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc_));
      
      CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
      CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
      if (this->has_bias_term()){
	CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc_));
      }
      CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
    }
    
    if (workspace_)
      CHECK_CUDA(cudaFree(workspace_));
  }

  template <typename Dtype>
  void CUDNNConvLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    //   std::vector<int> output_shape;
    //this->ComputeOutputShape(input_shape, output_shape);
    //this->outputs_[0]->Reshape(output_shape);


    CHECK_CUDNN(cudnnSetTensor4dDescriptorEx(input_desc_,
					     // cudnnTensorFormat,
					     cudnnDataType,
					     input_shape[0], //n
					     input_shape[1] / this->group_, //c
					     input_shape[2], //h
					     input_shape[3],  //w

					     input_shape[1] * input_shape[2] * input_shape[3],
					     input_shape[2] * input_shape[3],
					     input_shape[3],
					     1
					     ));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
					   cudnnDataType,
					   cudnnTensorFormat,
					   this->num_output_ / this->group_,
					   input_shape[1] / this->group_, //c 
					   this->kernel_size_h_,
					   this->kernel_size_w_
					   ));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
						this->pad_h_,
						this->pad_w_,      // padding
						this->stride_h_,
						this->stride_w_,   // stride
						this->dilation_h_,
						this->dilation_w_, // dilation  CUDNN 6.0
						CUDNN_CROSS_CORRELATION,
						cudnnDataType
						));
   
  
    std::vector<int> output_shape;
    this->ComputeOutputShape(input_shape, output_shape);    
    CHECK_CUDNN(cudnnSetTensor4dDescriptorEx(output_desc_,
					     //  cudnnTensorFormat,
					     cudnnDataType,
					     output_shape[0], //n
					     output_shape[1] / this->group_, //c
					     output_shape[2], //h
					     output_shape[3],  //w

					     output_shape[1] * output_shape[2] * output_shape[3],
					     output_shape[2] * output_shape[3],
					     output_shape[3],
					     1
					     ));

    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle_,
						    input_desc_,
						    filter_desc_,
						    conv_desc_,
						    output_desc_,
						    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
						    cudnn_workspace_limit_bytes,
						    &fwd_algo_
						    ));

   
	

    size_t size_in_bytes=0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_,
							input_desc_,
							filter_desc_,
							conv_desc_,
							output_desc_,
							fwd_algo_,
							&size_in_bytes
							));

    if (workspace_)
      CHECK_CUDA(cudaFree(workspace_));
    if (size_in_bytes!=0){
      CHECK_CUDA(cudaMalloc(&workspace_,size_in_bytes));
      workspace_size_in_bytes_ = size_in_bytes;
    }else{
      workspace_ = nullptr;
      workspace_size_in_bytes_ = 0;
      fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    }
    //std::cout<<"workspace size: "<<workspace_size_in_bytes_<<std::endl;

    //std::cout<<"CONV ALGO: "<<fwd_algo_<<std::endl;
    
    if (this->has_bias_term()){
      CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc_,
					     cudnnTensorFormat,
					     cudnnDataType,
					     1, //n
					     this->num_output_ / this->group_, //c
					     1, //h
					     1  //w
					     ));
    }

    this->outputs_[0]->Reshape(output_shape);
  }

  template <typename Dtype>
  void CUDNNConvLayer<Dtype>::ForwardComputation(){
    if (!this->CheckWeights()) return;

    
    Dtype alpha = Dtype(1);
    Dtype beta  = Dtype(0);
    const Dtype * input_data = this->inputs_[0]->gpu_data();
    Dtype * output_data = this->outputs_[0]->mutable_gpu_data();
    const Dtype * filter_weight = this->weights_[0]->gpu_data();

    int in_channels = this->inputs_[0]->shape(1);
    int weight_offset = this->num_output_ * in_channels *
      this->kernel_size_h_ * this->kernel_size_w_ / this->group_ / this->group_;
    
    int input_offset  = this->inputs_[0]->count(1) / this->group_;
    int output_offset = this->outputs_[0]->count(1) / this->group_;
    int bias_offset = this->num_output_ / this->group_;
    
    for (int g = 0; g < this->group_; g++){
      CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle_,
					  &alpha,
					  input_desc_,
					  input_data + input_offset * g,
					  filter_desc_,
					  filter_weight + weight_offset * g,
					  conv_desc_,
					  fwd_algo_,
					  workspace_,
					  workspace_size_in_bytes_,
					  &beta,
					  output_desc_,
					  output_data + output_offset * g
					  ));

      if (this->has_bias_term()) {
	Dtype b_alpha = Dtype(1);
	Dtype b_beta  = Dtype(1);
	const Dtype* bias_data = this->weights_[1]->gpu_data();
	CHECK_CUDNN(cudnnAddTensor(cudnn_handle_,
				   &b_alpha,
				   bias_desc_,
				   bias_data + bias_offset * g,
				   &b_beta,
				   output_desc_,
				   output_data + output_offset * g
				   ));
      }
    }
  }
  
  INSTANTIATE_CLASS(CUDNNConvLayer);
} // namespace facethink
