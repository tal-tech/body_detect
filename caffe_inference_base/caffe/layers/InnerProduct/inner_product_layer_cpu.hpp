#ifndef __FACETHINK_CAFFE_LAYERS_INNER_PRODUCT_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_INNER_PRODUCT_LAYER_CPU_HPP__

#include "caffe/layers/InnerProduct/inner_product_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class InnerProductLayer: public BaseInnerProductLayer<Dtype> {
  public:
    explicit InnerProductLayer(int num_output,
			       bool has_bias_term = true,
			       int axis = 1,
			       bool transpose = false,
			       const std::string& name = "")
      :BaseInnerProductLayer<Dtype>(num_output,
				    has_bias_term,
				    axis,
				    transpose,
				    name) {
    }

    virtual void ForwardComputation();

  private:
    DISABLE_COPY_AND_ASSIGN(InnerProductLayer);
  }; // class InnerProductLayer

} // namespace


#endif
