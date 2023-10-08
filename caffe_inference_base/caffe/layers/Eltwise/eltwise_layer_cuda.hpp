#ifndef __FACETHINK_CAFFE_LAYERS_ELTWISE_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_ELTWISE_LAYER_CUDA_HPP__

#include "caffe/layers/Eltwise/eltwise_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDAEltwiseLayer: public BaseEltwiseLayer<Dtype> {
  public:
    explicit CUDAEltwiseLayer(EltwiseOp op,
			      const std::vector<Dtype> coeffs,
			      const std::string& name="")
      :BaseEltwiseLayer<Dtype>(op,
			       coeffs,
			       name),
      cublas_handle_(nullptr) {

      InitLayer();
    }

    ~CUDAEltwiseLayer();

    virtual void InitLayer();
    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "Eltwise(CUDA)";
    }

  private:
    cublasHandle_t cublas_handle_;

    DISABLE_COPY_AND_ASSIGN(CUDAEltwiseLayer);
  }; // class CUDAEltwiseLayer

} // namespace facethink

#endif
