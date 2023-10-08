#include "caffe/layers/LRN/lrn_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void LRNLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    
    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    this->outputs_[0]->Reshape(input_shape);

    scale_.Reshape(input_shape);
  }

  template <typename Dtype>
  void LRNLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    Dtype* scale_data = scale_.mutable_cpu_data();
    // start with the constant value
    for (int i = 0; i < scale_.count(); ++i) {
      scale_data[i] = this->k_;
    }

    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    int num      = input_shape[0];
    int channels = input_shape[1];
    int height   = input_shape[2];
    int width    = input_shape[3];
    int pre_pad  = (this->size_ - 1) / 2;
    
    Blob<Dtype> padded_square(std::vector<int> { 1, channels + this->size_ - 1, height, width });
    Dtype* padded_square_data = padded_square.mutable_cpu_data(); 
    caffe_cpu_set(padded_square.count(), Dtype(0), padded_square_data);
    
    Dtype alpha_over_size = this->alpha_ / this->size_;
    // go through the images
    for (int n = 0; n < num; ++n) {
      // compute the padded square
      caffe_cpu_sqr(channels * height * width,
		    input_data + this->inputs_[0]->offset(n),
		    padded_square_data + padded_square.offset(0, pre_pad));
      // Create the first channel scale
      for (int c = 0; c < this->size_; ++c) {
	caffe_cpu_axpy<Dtype>(height * width, alpha_over_size,
			      padded_square_data + padded_square.offset(0, c),
			      scale_data + scale_.offset(n, 0));
      }
      for (int c = 1; c < channels; ++c) {
	// copy previous scale
	caffe_cpu_copy<Dtype>(height * width,
			      scale_data + scale_.offset(n, c - 1),
			      scale_data + scale_.offset(n, c));
	// add head
	caffe_cpu_axpy<Dtype>(height * width, alpha_over_size,
			      padded_square_data + padded_square.offset(0, c + this->size_ - 1),
			      scale_data + scale_.offset(n, c));
	// subtract tail
	caffe_cpu_axpy<Dtype>(height * width, -alpha_over_size,
			      padded_square_data + padded_square.offset(0, c - 1),
			      scale_data + scale_.offset(n, c));
      }
    }

    // In the end, compute output
    caffe_cpu_powx<Dtype>(scale_.count(), scale_data, -this->beta_, output_data);
    caffe_cpu_mul<Dtype>(scale_.count(), output_data, input_data, output_data);
  }

  INSTANTIATE_CLASS(LRNLayer);
}
