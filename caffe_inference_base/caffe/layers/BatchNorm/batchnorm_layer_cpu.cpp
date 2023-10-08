#include "caffe/layers/BatchNorm/batchnorm_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void BatchNormLayer<Dtype>::ForwardComputation() {
    this->CheckWeights();

    if (!this->use_global_stats_) {
      BOOST_LOG_TRIVIAL(fatal)<< "BatchNormLayer: Must use pre-trained mean and variances in the TEST phase.";
      return;
    }
    
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();

    const std::vector<int> input_shape = this->inputs_[0]->shape();
    const int input_count = this->inputs_[0]->count();
    
    int num = input_shape[0];
    int spatial_dim = input_count /(this->channels_ * input_shape[0]);

    if (this->inputs_[0] != this->outputs_[0]) {
      caffe_cpu_copy(input_count, input_data, output_data);
    }


    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->weights_[2]->cpu_data()[0] == 0 ?
      0 : 1 / this->weights_[2]->cpu_data()[0];
      
    caffe_cpu_scale(this->variance_.count(),
		    scale_factor,
		    this->weights_[0]->cpu_data(),
		    this->mean_.mutable_cpu_data());
    caffe_cpu_scale(this->variance_.count(),
		    scale_factor,
		    this->weights_[1]->cpu_data(),
		    this->variance_.mutable_cpu_data());
   
    // subtract mean
    caffe_cpu_gemm<Dtype>(CblasNoTrans,
			  CblasNoTrans,
			  num,
			  this->channels_,
			  1,
			  Dtype(1.),
			  this->batch_sum_multiplier_.cpu_data(),
			  this->mean_.cpu_data(),
			  Dtype(0.),
			  this->num_by_chans_.mutable_cpu_data());
    
    caffe_cpu_gemm<Dtype>(CblasNoTrans,
			  CblasNoTrans,
			  this->channels_ * num,
			  spatial_dim,
			  1,
			  Dtype(-1.),
			  this->num_by_chans_.cpu_data(),
			  this->spatial_sum_multiplier_.cpu_data(),
			  Dtype(1.),
			  output_data);

    // normalize variance
    caffe_cpu_add_scalar(this->variance_.count(),
			 this->eps_,
			 this->variance_.mutable_cpu_data());
    
    caffe_cpu_powx(this->variance_.count(),
		   this->variance_.cpu_data(),
		   Dtype(0.5),
		   this->variance_.mutable_cpu_data());

    // replicate variance to input size
    caffe_cpu_gemm<Dtype>(CblasNoTrans,
			  CblasNoTrans,
			  num, this->channels_, 1,
			  Dtype(1.0),
			  this->batch_sum_multiplier_.cpu_data(),
			  this->variance_.cpu_data(),
			  Dtype(0.),
			  this->num_by_chans_.mutable_cpu_data());
    
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			  this->channels_ * num, spatial_dim, 1,
			  Dtype(1.),
			  this->num_by_chans_.cpu_data(),
			  this->spatial_sum_multiplier_.cpu_data(),
			  Dtype(0.),
			  this->temp_.mutable_cpu_data());
    
    caffe_cpu_div(this->temp_.count(), output_data, this->temp_.cpu_data(), output_data);
  }
  
  INSTANTIATE_CLASS(BatchNormLayer);
  
} // namespace facethink
