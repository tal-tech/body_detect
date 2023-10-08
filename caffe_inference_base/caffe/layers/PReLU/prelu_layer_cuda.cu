#include <algorithm>
#include <vector>

#include "caffe/layers/PReLU/prelu_layer_cuda.hpp"

namespace facethink {

// CUDA kernele for forward
template <typename Dtype>
__global__ void PReLUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* slope_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}



template <typename Dtype>
void CUDAPReluLayer<Dtype>::ForwardComputation() {

  const Dtype* input_data = this->inputs_[0]->gpu_data();
  Dtype* output_data = this->outputs_[0]->mutable_gpu_data();
  const int count = this->inputs_[0]->count();

  const int dim = this->inputs_[0]->count(2);
  const int channels = this->inputs_[0]->channels();
  const Dtype* slope_data = this->weights_[0]->gpu_data();

  const int div_factor = this->channel_shared_ ? channels : 1;


  // NOLINT_NEXT_LINE(whitespace/operators)
  PReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, input_data, output_data, slope_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(CUDAPReluLayer);


}  // namespace caffe
