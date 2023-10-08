#include "caffe/layers/DepthWiseConvolution/depthwise_conv_layer_cuda.hpp"

namespace facethink {

  template <typename Dtype>
  void CUDADepthwiseConvLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    this->CheckWeights();

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    std::vector<int> output_shape;
    this->ComputeOutputShape(input_shape, output_shape);
    this->outputs_[0]->Reshape(output_shape);
  }



  template <typename Dtype>
  __global__ void ConvForward(const int nthreads,
                              const Dtype* const bottom_data,
                              const int num, const int channels,
                              const int height, const int width,
                              const int conved_height, const int conved_width,
                              const int kernel_h, const int kernel_w,
                              const int stride_h, const int stride_w,
                              const int pad_h, const int pad_w,
                              Dtype* const top_data,
                              const Dtype* const weight,
                              const Dtype* const bias,
                              const bool bias_term_) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int pw = index % conved_width;
      const int ph = (index / conved_width) % conved_height;
      const int c = (index / conved_width / conved_height) % channels;
      const int n = index / conved_width / conved_height / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = min(hstart + kernel_h, height + pad_h);
      int wend = min(wstart + kernel_w, width + pad_w);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
      hend = min(hend, height);
      wend = min(wend, width);
      Dtype aveval = 0;
      const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
      const Dtype* const weight_slice =
        weight + c * kernel_h * kernel_w;

      int khstart=hend<kernel_h?kernel_h-hend:0;
      int kwstart=wend<kernel_w?kernel_w-wend:0;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_slice[h * width + w]*weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
        }
      }
      if(bias_term_) {
        aveval+=bias[c];
      }
      top_data[index] = aveval;
    }
  }

  template <typename Dtype>
  void CUDADepthwiseConvLayer<Dtype>::ForwardComputation() {
    this->CheckWeights();

    const Dtype* weight_data = this->weights_[0]->gpu_data();
    const Dtype* input_data = this->inputs_[0]->gpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_gpu_data();

    const int count = this->outputs_[0]->count();

    const std::vector<int>&  input_shape = this->inputs_[0]->shape();
    const int input_num = input_shape[0];
    const int input_channels = input_shape[1];
		const int input_height = input_shape[2];
		const int input_width = input_shape[3];


    const std::vector<int>&  output_shape = this->outputs_[0]->shape();
    const int conved_height = output_shape[2];
    const int conved_weight = output_shape[3];

    if (this->has_bias_term()) {
      const Dtype* bias_data = this->weights_[1]->gpu_data();
      ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              count,
                                                                              input_data,
                                                                              input_num, input_channels, input_height, input_width,
                                                                              conved_height, conved_weight,
                                                                              this->kernel_size_h_, this->kernel_size_w_,
                                                                              this->stride_h_, this->stride_w_,
                                                                              this->pad_h_, this->pad_w_,
                                                                              output_data,
                                                                              weight_data,
                                                                              bias_data, true);
    } else {
      Dtype* bias_data = 0;
      ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                                                                              count,
                                                                              input_data,
                                                                              input_num, input_channels, input_height, input_width,
                                                                              conved_height, conved_weight,
                                                                              this->kernel_size_h_, this->kernel_size_w_,
                                                                              this->stride_h_, this->stride_w_,
                                                                              this->pad_h_, this->pad_w_,
                                                                              output_data,
                                                                              weight_data,
                                                                              bias_data, false);
    }
  }

  INSTANTIATE_CLASS(CUDADepthwiseConvLayer);
}
