#ifndef __FACETHINK_CAFFE_LAYERS_CONV_LAYER_CUDNN_HPP__
#define __FACETHINK_CAFFE_LAYERS_CONV_LAYER_CUDNN_HPP__

#include "caffe/layers/Convolution/conv_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDNNConvLayer: public BaseConvLayer<Dtype> {
  public:
    explicit CUDNNConvLayer(int num_output,
			    int kernel_size_h, int kernel_size_w,
			    int pad_h = 0, int pad_w = 0,
			    int stride_h = 1, int stride_w = 1,
			    int dilation_h = 1, int dilation_w = 1,
			    bool bias_term = true,
			    int group = 1,
			    const std::string& name = "")
      :BaseConvLayer<Dtype>(num_output,
			    kernel_size_h, kernel_size_w,
			    pad_h, pad_w,
			    stride_h, stride_w,
			    dilation_h, dilation_w,
			    bias_term, group, name),
      cudnn_handle_(nullptr),
      workspace_size_in_bytes_(0),
      workspace_(nullptr) {

      InitLayer();
    }

    ~CUDNNConvLayer();

    virtual void InitLayer();
    virtual void ForwardShape();
    virtual void ForwardComputation();

    virtual inline std::string type() const { return "Convolution(CUDNN)"; }

  private:
    cudnnHandle_t cudnn_handle_;

    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;

    cudnnFilterDescriptor_t filter_desc_;
    cudnnTensorDescriptor_t bias_desc_;

    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnConvolutionFwdAlgo_t fwd_algo_;

    size_t workspace_size_in_bytes_;  // size of underlying storage
    //void *workspaceData;    // underlying storage
    void * workspace_;  // underlying storage

    DISABLE_COPY_AND_ASSIGN(CUDNNConvLayer);
  }; // class CUDNNConvLayer

} // namespace facethink

#endif
