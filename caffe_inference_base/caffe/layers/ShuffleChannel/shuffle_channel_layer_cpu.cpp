#include "caffe/layers/ShuffleChannel/shuffle_channel_layer_cpu.hpp"
#include "caffe/util/math_func.hpp"

namespace facethink {

  template <typename Dtype>
  void ShuffleChannelLayer<Dtype>::Resize_cpu(Dtype *output,
					      const Dtype *input,
					      int group_row,
					      int group_column,
					      int len) {
    for (int i = 0; i < group_row; ++i) { // 2
      for(int j = 0; j < group_column ; ++j) { // 3
	const Dtype* p_i = input + (i * group_column + j ) * len;
	Dtype* p_o = output + (j * group_row + i ) * len;
	caffe_cpu_copy(len, p_i, p_o);
      }
    }
  }

  template <typename Dtype>
  void ShuffleChannelLayer<Dtype>::ForwardComputation() {
    const Dtype* intput_data = this->inputs_[0]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    
    for(int n = 0; n < this->num_; ++n) {
      Resize_cpu(output_data + n * this->feature_map_size_,
		 intput_data + n * this->feature_map_size_,
		 this->group_row_,
		 this->group_column_,
		 this->sp_sz_);
    }
  }

  INSTANTIATE_CLASS(ShuffleChannelLayer);
}
