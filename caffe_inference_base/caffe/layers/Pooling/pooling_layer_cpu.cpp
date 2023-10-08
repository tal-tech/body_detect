#include "caffe/layers/Pooling/pooling_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"
#include <cfloat>

namespace facethink {

  template <typename Dtype>
  void PoolingLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    this->CheckMode();

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    std::vector<int> output_shape;
    this->ComputeOutputShape(input_shape, output_shape);

    this->outputs_[0]->Reshape(output_shape);
  }


  template <typename Dtype>
  void PoolingLayer<Dtype>::ForwardComputation() {
    const Dtype * input_data = this->inputs_[0]->cpu_data();
    Dtype * output_data = this->outputs_[0]->mutable_cpu_data();
    int output_count = this->outputs_[0]->count();

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    const std::vector<int>& output_shape = this->outputs_[0]->shape();

    int num    = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width  = input_shape[3];
    int pooled_height = output_shape[2];
    int pooled_width  = output_shape[3];
    
    if (this->mode_ == POOLING_MAX) {
      caffe_cpu_set(output_count, Dtype(-FLT_MAX), output_data);
      // The main loop
      for (int n = 0; n < num; ++n) {
	for (int c = 0; c < channels; ++c) {
	  for (int ph = 0; ph < pooled_height; ++ph) {
	    for (int pw = 0; pw < pooled_width; ++pw) {
	      int hstart = ph * this->stride_h_ - this->pad_h_;
	      int wstart = pw * this->stride_w_ - this->pad_w_;
	      int hend = std::min(hstart + this->kernel_size_h_, height);
	      int wend = std::min(wstart + this->kernel_size_w_, width );
	      hstart = std::max(hstart, 0);
	      wstart = std::max(wstart, 0);
	      const int pool_index = ph * pooled_width + pw;
	      for (int h = hstart; h < hend; ++h) {
		for (int w = wstart; w < wend; ++w) {
		  const int index = h * width + w;
		  if (input_data[index] > output_data[pool_index]) {
		    output_data[pool_index] = input_data[index];
		  }
		}
	      }
	    }
	  }
	  // compute offset
	  input_data += height * width;
	  output_data += pooled_height * pooled_width;
	}
      }       
    } else if (this->mode_ = POOLING_AVE) {
      caffe_cpu_set(output_count, Dtype(0.), output_data);
      // The main loop
      for (int n = 0; n < num; ++n) {
	for (int c = 0; c < channels; ++c) {
	  for (int ph = 0; ph < pooled_height; ++ph) {
	    for (int pw = 0; pw < pooled_width; ++pw) {
	      int hstart = ph * this->stride_h_ - this->pad_h_;
	      int wstart = pw * this->stride_w_ - this->pad_w_;
	      int hend = std::min(hstart + this->kernel_size_h_, height + this->pad_h_);
	      int wend = std::min(wstart + this->kernel_size_w_, width  + this->pad_w_);
	      int pool_size = (hend - hstart) * (wend - wstart);
	      hstart = std::max(hstart, 0);
	      wstart = std::max(wstart, 0);
	      hend = std::min(hend, height);
	      wend = std::min(wend, width);
	      for (int h = hstart; h < hend; ++h) {
		for (int w = wstart; w < wend; ++w) {
		  output_data[ph * pooled_width + pw] +=
                    input_data[h * width + w];
		}
	      }
	      output_data[ph * pooled_width + pw] /= pool_size;
	    }
	  }
	  // compute offset
	  input_data += height * width;
	  output_data += pooled_height * pooled_width;
	}
      }

    } else {

    }
  }

  INSTANTIATE_CLASS(PoolingLayer);

} // namespace facethink
