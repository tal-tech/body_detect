#include "caffe/layers/Convolution/conv_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseConvLayer<Dtype>::ComputeOutputShape(const std::vector<int>& input_shape,
						std::vector<int>& output_shape) {
    output_shape.clear();
    output_shape.push_back(input_shape[0]); // num
    output_shape.push_back(this->num_output_);   // c

    const int input_dim_h = input_shape[2];
    const int input_dim_w = input_shape[3];
    const int kernel_extent_h = dilation_h_ * (kernel_size_h_ - 1) + 1;
    const int kernel_extent_w = dilation_w_ * (kernel_size_w_ - 1) + 1;

    const int output_dim_h = (input_dim_h + 2 * pad_h_ - kernel_extent_h) / stride_h_ + 1;
    const int output_dim_w = (input_dim_w + 2 * pad_w_ - kernel_extent_w) / stride_w_ + 1;

    output_shape.push_back(output_dim_h);
    output_shape.push_back(output_dim_w);
  }

  INSTANTIATE_CLASS(BaseConvLayer);

} // namespace facethink
