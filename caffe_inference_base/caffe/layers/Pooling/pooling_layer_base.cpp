#include "caffe/layers/Pooling/pooling_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BasePoolingLayer<Dtype>::ComputeOutputShape(const std::vector<int>& input_shape,
						   std::vector<int>& output_shape) {
    int height = input_shape[2];
    int width = input_shape[3];
    
    if (global_pooling_){  
      kernel_size_h_ = input_shape[2];
      kernel_size_w_ = input_shape[3];
    }
 
    int pooled_height =
      static_cast<int>(ceil(static_cast<float>(height + 2*pad_h_ - kernel_size_h_) / stride_h_)) + 1;
    int pooled_width =
      static_cast<int>(ceil(static_cast<float>(width + 2*pad_w_ - kernel_size_w_) / stride_w_)) + 1;

    if (pad_h_ || pad_w_) {
      // If we have padding, ensure that the last pooling starts strictly
      // inside the image (instead of at the padding); otherwise clip the last.
      if ((pooled_height-1)*stride_h_ >= height+pad_h_){
	--pooled_height;
      }
      if ((pooled_width -1)*stride_w_ >= width +pad_w_){
	--pooled_width;
      }
      if ((pooled_height-1)*stride_h_ >= height+pad_h_ ||
	  (pooled_width -1)*stride_w_ >= width +pad_w_){
	BOOST_LOG_TRIVIAL(error)<<"PoolingLayer: Invalid Pooling size (kernel, pad, stride)";
      }
    }
    
    output_shape = std::vector<int> { input_shape[0], input_shape[1], pooled_height, pooled_width };
  }

  INSTANTIATE_CLASS(BasePoolingLayer);
} // namespace facethink
