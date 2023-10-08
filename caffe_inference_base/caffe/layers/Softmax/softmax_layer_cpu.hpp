#ifndef __FACETHINK_CAFFE_LAYERS_SOFTMAX_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_SOFTMAX_LAYER_CPU_HPP__


#include "caffe/layers/Softmax/softmax_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class SoftmaxLayer: public BaseSoftmaxLayer<Dtype> {
  public:
    explicit SoftmaxLayer(int softmax_axis,
			  const std::string& name = "")
      :BaseSoftmaxLayer<Dtype>(softmax_axis, name) {
    }


    virtual void ForwardShape();
    virtual void ForwardComputation();

  private:
    int outer_num_;
    int inner_num_;

    int softmax_axis_canonical_;

    /// sum_multiplier is used to carry out sum using BLAS
    Blob<Dtype> sum_multiplier_;
    /// scale is an intermediate Blob to hold temporary results.
    Blob<Dtype> scale_;

    DISABLE_COPY_AND_ASSIGN(SoftmaxLayer);
  }; // class SoftmaxLayer

} // namespace facethink




#endif
