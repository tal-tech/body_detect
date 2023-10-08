#include "caffe/layers/Normalize/normalize_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void NormalizeLayer<Dtype>::ForwardComputation() {
    const std::vector<int> input_shape = this->inputs_[0]->shape();
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    const Dtype* scale_data = this->weights_[0]->cpu_data();
    Dtype* buffer_data = this->buffer_.mutable_cpu_data();
    Dtype* norm_data = this->norm_.mutable_cpu_data();
    // add eps to avoid overflow
    caffe_cpu_set<Dtype>(this->norm_.count(), this->eps_, norm_data);

    const Dtype* sum_channel_multiplier = this->sum_channel_multiplier_.cpu_data();
    const Dtype* sum_spatial_multiplier = this->sum_spatial_multiplier_.cpu_data();
    
    int num = input_shape[0];
    int dim = input_shape[1] * input_shape[2] * input_shape[3];
    int spatial_dim = input_shape[2] * input_shape[3];
    int channels = input_shape[1]; 

    for (int n = 0; n < num; ++n) {
      caffe_cpu_sqr<Dtype>(dim, input_data, buffer_data);
      if (this->across_spatial_) {
	norm_data[n] = pow(caffe_cpu_asum<Dtype>(dim, buffer_data) + this->eps_,
	 		   Dtype(0.5));
	caffe_cpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), input_data,
	 		       output_data);
      } else {
        caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
	 		      buffer_data, sum_channel_multiplier, Dtype(1),
	 		      norm_data);
	// compute norm
	caffe_cpu_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);

	
	// scale the layer
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
	 		      1, Dtype(1), sum_channel_multiplier, norm_data,
	 		      Dtype(0), buffer_data);
	caffe_cpu_div<Dtype>(dim, input_data, buffer_data, output_data);
	norm_data += spatial_dim;
      }
      
      // scale the output
      if (this->channel_shared_) {
	caffe_cpu_scal<Dtype>(dim, scale_data[0], output_data);
      } else {
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
			      1, Dtype(1), scale_data, sum_spatial_multiplier,
			      Dtype(0),
			      buffer_data);
	caffe_cpu_mul<Dtype>(dim, output_data, buffer_data, output_data);
      }
      input_data += dim;
      output_data += dim;
    }
  }

  INSTANTIATE_CLASS(NormalizeLayer);
} // namespace facethink
