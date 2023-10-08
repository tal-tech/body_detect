#ifndef __FACETHINK_CAFFE_LAYERS_CONV_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_CONV_LAYER_CUDA_HPP__

#include "caffe/layers/Convolution/conv_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDAConvLayer: public BaseConvLayer<Dtype> {
  public:
    explicit CUDAConvLayer(int num_output,
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
      cublas_handle_(nullptr) {

      InitLayer();
    }

    ~CUDAConvLayer();

    virtual void InitLayer();
    virtual void ForwardShape();
    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "Convolution(CUDA)";
    }

  private:
    void forward_gpu_gemm(const Dtype* input, const Dtype* weights,
			  Dtype* output);
    void conv_im2col_gpu(const Dtype* data, Dtype* col_buff);
    void forward_gpu_bias(Dtype* output, const Dtype* bias);

  private:
    cublasHandle_t cublas_handle_;

    bool is_1x1_;
    int num_spatial_axes_, channel_axis_,  first_spatial_axis_;

    int conv_in_channels_, conv_in_height_, conv_in_width_;
    int conv_out_channels_;

    int kernel_dim_;
    int conv_out_spatial_dim_;
    int out_spatial_dim_;

    int weight_offset_;

    std::shared_ptr<Blob<Dtype> > col_buffer_;
    int col_offset_;

    int output_offset_;

    Blob<Dtype> bias_multiplier_;

    DISABLE_COPY_AND_ASSIGN(CUDAConvLayer);

  }; // class CUDAConvLayer


} // namespace facethink

#endif
