#ifndef __FACETHINK_CAFFE_LAYERS_LRN_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_LRN_LAYER_CPU_HPP__

#include "caffe/layers/LRN/lrn_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class LRNLayer: public BaseLRNLayer<Dtype> {
  public:
    explicit LRNLayer(int size,
		      Dtype alpha,
		      Dtype beta,
		      Dtype k,
		      const std::string& name="")
      :BaseLRNLayer<Dtype> (size, alpha, beta, k, name) {
    }

    virtual void ForwardShape();
    virtual void ForwardComputation();

  private:
    Blob<Dtype> scale_;
  }; // class LRNLayer

} // namespace facethink

#endif
