#ifndef __FACETHINK_CAFFE_LAYERS_SCALE_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_SCALE_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseScaleLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseScaleLayer(int axis = 1,
			    int num_axes = 1,
			    bool bias_term = false,
			    const std::string& name="")
      :BaseLayer<Dtype>(name),
      axis_(axis),
      num_axes_(num_axes),
      bias_term_(bias_term) {
    }
    
    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;
   
    virtual inline std::string type() const {
      return "Scale";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream <<"("<<this->name_<<")"
	     <<", axis: "<<axis_
	     <<", num_axes: "<<num_axes_
	     <<", bias_term: "<< (bias_term_? "True":"False");
      return stream.str();
    }

    /// set and output parameters
    virtual inline bool has_weights() const { return true; }
    inline int axis() const { return axis_; }
    inline int num_axes() const { return num_axes_; }
    inline bool bias_term() const { return bias_term_; }

  protected:
    virtual inline bool CheckBlobs() const {
      if (!(this->inputs_.size() == 1 || this->inputs_.size() == 2)
	  || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"BaseScaleLayer: only accept one or two input blob and one output blob.";
	return false;
      }
      return true;
    }

    inline bool CheckWeights() const {
      if (this->weights_.empty()) {
	BOOST_LOG_TRIVIAL(error)<<"BaseScaleLayer: Invalid weights or weights have not been imported.";
	return false;
      }
      return true;
    }
    
  protected:
    int axis_;
    int num_axes_;
    bool bias_term_;

    int axis_valid_;
    int outer_dim_, scale_dim_, inner_dim_, bias_dim_;

    Blob<Dtype> bias_multiplier_;
    
    DISABLE_COPY_AND_ASSIGN(BaseScaleLayer);
  }; // class BaseScaleLayer


} // namespace facethink


#endif
