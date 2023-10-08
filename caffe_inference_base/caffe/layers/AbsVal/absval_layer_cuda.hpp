#ifndef __FACETHINK_CAFFE_LAYERS_ABSVAL_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_ABSVAL_LAYER_CUDA_HPP__

#include "caffe/layers/AbsVal/absval_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDAAbsValLayer: public BaseAbsValLayer<Dtype> {
  public:
    explicit CUDAAbsValLayer(const std::string& name="")
      :BaseAbsValLayer<Dtype>(name) {
    }

    virtual void ForwardComputation();

    virtual inline std::string type() const { return "AbsVal(CUDNN)"; }

  private:

    DISABLE_COPY_AND_ASSIGN(CUDAAbsValLayer);
  }; // class CUDAAbsValLayer

}


#endif
