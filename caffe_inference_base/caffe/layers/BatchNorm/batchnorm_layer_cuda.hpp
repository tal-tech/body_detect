#ifndef __FACETHINK_CAFFE_LAYERS_BATCHNORM_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_BATCHNORM_LAYER_CUDA_HPP__

#include "caffe/layers/BatchNorm/batchnorm_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDABatchNormLayer: public BaseBatchNormLayer<Dtype> {
  public:
    explicit CUDABatchNormLayer(bool use_global_stats,
				Dtype moving_average_fraction = Dtype(.999),
				Dtype eps = Dtype(1e-5),
				const std::string& name="")
      :BaseBatchNormLayer<Dtype>(use_global_stats,
				 moving_average_fraction,
				 eps,
				 name),
      cublas_handle_(nullptr) {

      InitLayer();
    }

    ~CUDABatchNormLayer();
    virtual void InitLayer();

    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "BatchNorm(CUDA)";
    }

  private:
    cublasHandle_t cublas_handle_;

    DISABLE_COPY_AND_ASSIGN(CUDABatchNormLayer);
  }; // class CUDABatchNormLayer

} // namespace facethink


#endif
