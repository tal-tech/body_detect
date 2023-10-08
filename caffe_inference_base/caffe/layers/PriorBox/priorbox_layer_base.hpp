#ifndef __FACETHINK_CAFFE_LAYERS_PRIORBOX_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_PRIORBOX_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BasePriorBoxLayer: public BaseLayer<Dtype> {
  public:
    explicit BasePriorBoxLayer(Dtype min_size, Dtype max_size,
			       const std::vector<Dtype>& aspect_ratios,
			       const std::vector<Dtype>& variances,
			       bool flip,
			       bool clip,
			       int img_h, int img_w,
			       Dtype step_h, Dtype step_w,
			       Dtype offset,
			       const std::string& name="")
      :BaseLayer<Dtype>(name),
      min_size_(min_size), max_size_(max_size),
      aspect_ratios_(aspect_ratios),
      variances_(variances),
      flip_(flip), clip_(clip),
      img_h_(img_h), img_w_(img_w),
      step_h_(step_h), step_w_(step_w),
      offset_(offset) {

      InitLayer();
    }

    virtual void InitLayer();
    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;

    virtual inline std::string type() const {
      return "PriorBox";
    }
    
    virtual inline std::string param_string() const{
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<", min_size: "<<min_size_
	    <<", max_size: "<<max_size_;
      
      stream<<", aspect_ratios: [";
      for (size_t i = 0; i < aspect_ratios_.size(); ++i) {
	stream<<aspect_ratios_.at(i)<<",";
      }
      stream<<"]";

      stream<<", flip: "<<(flip_? "True":"False")
	    <<", clip: "<<(clip_? "True":"False");
      
      stream<<", variances: [";
      for (size_t i = 0; i < variances_.size(); ++i) {
	stream<<variances_.at(i)<<",";
      }
      stream<<"]";

      stream<<", img_size: ["<<img_h_<<","<<img_w_<<"]";
      stream<<", step: ["<<step_h_<<","<<step_w_<<"]";
      stream<<", offset: "<<offset_;
      
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 2 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"PriorBoxLayer: only accept two input blobs and one output blob.";
	return false;
      }
      return true;
    }
    
  protected:
    Blob<Dtype> aspect_ratios_blob_;
    Blob<Dtype> variances_blob_;

    Dtype min_size_, max_size_;
    std::vector<Dtype> aspect_ratios_, variances_;

    bool flip_, clip_;
    int num_priors_;

    int img_h_, img_w_;
    Dtype step_h_, step_w_;
    Dtype offset_;
    
    DISABLE_COPY_AND_ASSIGN(BasePriorBoxLayer);
  }; // class BasePriorBoxLayer

} // namespace facethink

#endif
