#ifndef __FACETHINK_CAFFE_LAYERS_PRIORBOX_LAYER_CPU_HPP__
#define __FACETHINK_CAFFE_LAYERS_PRIORBOX_LAYER_CPU_HPP__

#include "caffe/layers/PriorBox/priorbox_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class PriorBoxLayer: public BasePriorBoxLayer<Dtype> {
  public:
    explicit PriorBoxLayer(Dtype min_size, Dtype max_size,
			   const std::vector<Dtype>& aspect_ratios,
			   const std::vector<Dtype>& variances,
			   bool flip,
			   bool clip,
			   int img_h, int img_w,
			   Dtype step_h, Dtype step_w,
			   Dtype offset,
			   const std::string& name="")
      :BasePriorBoxLayer<Dtype>(min_size, max_size,
				aspect_ratios, variances,
				flip, clip,
				img_h, img_w,
				step_h, step_w,
				offset,
				name) {
    }

    virtual void ForwardComputation();


    DISABLE_COPY_AND_ASSIGN(PriorBoxLayer);
  }; // class PriorBoxLayer

} // namespace facethink


#endif
