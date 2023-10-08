#ifndef __FACETHINK_CAFFE_LAYERS_DROPOUT_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_DROPOUT_LAYER_CUDA_HPP__

#include "caffe/layers/Dropout/dropout_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDADropoutLayer: public BaseDropoutLayer<Dtype> {
  public:
    explicit CUDADropoutLayer(Dtype dropout_ratio,
			      const std::string& name="")
      :BaseDropoutLayer<Dtype>(dropout_ratio, name) {
    }

    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "Dropout(CUDA)";
    }

  }; // CUDADropoutLayer

}

#endif
