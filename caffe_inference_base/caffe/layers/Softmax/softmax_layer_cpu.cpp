#include "caffe/layers/Softmax/softmax_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void SoftmaxLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();

    softmax_axis_canonical_ = this->inputs_[0]->CanonicalAxisIndex(this->softmax_axis_);
    outer_num_ = this->inputs_[0]->count(0, softmax_axis_canonical_);
    inner_num_ = this->inputs_[0]->count(softmax_axis_canonical_ + 1);

    sum_multiplier_.Reshape( std::vector<int>{ this->inputs_[0]->shape(softmax_axis_canonical_) } );
    Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
    caffe_cpu_set(sum_multiplier_.count(), Dtype(1), multiplier_data);

    std::vector<int> scale_dims = this->inputs_[0]->shape();
    scale_dims[softmax_axis_canonical_] = 1;
    scale_.Reshape(scale_dims);
    
    this->outputs_[0]->Reshape(this->inputs_[0]->shape());
  }

  template <typename Dtype>
  void SoftmaxLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    Dtype* scale_data = scale_.mutable_cpu_data();
    
    int channels = this->inputs_[0]->shape(softmax_axis_canonical_);
    int dim = this->inputs_[0]->count() / outer_num_;
    caffe_cpu_copy(this->inputs_[0]->count(), input_data, output_data);
    // We need to subtract the max to avoid numerical issues, compute the exp,
    // and then normalize.
    for (int i = 0; i < outer_num_; ++i) {
      // initialize scale_data to the first plane
      caffe_cpu_copy(inner_num_, input_data + i * dim, scale_data);
      for (int j = 0; j < channels; j++) {
	for (int k = 0; k < inner_num_; k++) {
	  scale_data[k] = std::max(scale_data[k],
				   input_data[i * dim + j * inner_num_ + k]);
	}
      }
      // subtraction
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
			    1, -1., sum_multiplier_.cpu_data(), scale_data, 1., output_data);
      // exponentiation
      caffe_cpu_exp<Dtype>(dim, output_data, output_data);
      // sum after exp
      caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
			    output_data, sum_multiplier_.cpu_data(), 0., scale_data);
      // division
      for (int j = 0; j < channels; j++) {
	caffe_cpu_div(inner_num_, output_data, scale_data, output_data);
	output_data += inner_num_;
      }
    }
  }

  INSTANTIATE_CLASS(SoftmaxLayer);
} // namespace facethink
