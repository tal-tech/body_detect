#include "caffe/layers/PReLU/prelu_layer_cpu.hpp"

namespace facethink {

  template <typename Dtype>
  void PReluLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    const int count = this->inputs_[0]->count();

	const int dim = this->inputs_[0]->count(2);
	const int channels = this->inputs_[0]->channels();
	const Dtype* slope_data = this->weights_[0]->cpu_data();

	const int div_factor = this->channel_shared_ ? channels : 1;

	for (int i = 0; i < count; ++i) {
		int c = (i / dim) % channels / div_factor;
		output_data[i] = std::max(input_data[i], Dtype(0)) + slope_data[c] * std::min(input_data[i], Dtype(0));
	}

  }

  INSTANTIATE_CLASS(PReluLayer);
} // namespace facethink
