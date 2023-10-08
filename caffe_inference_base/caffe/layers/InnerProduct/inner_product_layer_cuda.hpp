#include "caffe/layers/InnerProduct/inner_product_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDAInnerProductLayer: public BaseInnerProductLayer<Dtype> {
  public:
    explicit CUDAInnerProductLayer(int num_output,
				   bool has_bias_term = true,
				   int axis = 1,
				   bool transpose = false,
				   const std::string& name = "")
      :BaseInnerProductLayer<Dtype>(num_output,
				    has_bias_term,
				    axis,
				    transpose,
				    name),
      cublas_handle_(nullptr) {

      InitLayer();
    }

    void InitLayer();
    ~CUDAInnerProductLayer();

    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "InnerProduct(CUDA)";
    }

  private:
    cublasHandle_t cublas_handle_;

    DISABLE_COPY_AND_ASSIGN(CUDAInnerProductLayer);
  }; // class CUDAInnerProductLayer

} // namespace facethink
