#ifndef __FACETHINK_CAFFE_LAYERS_RELU_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_RELU_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {
  
  template <typename Dtype>
  class BaseReluLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseReluLayer(bool inplace = true,
			   Dtype negative_slope = 0,
			   const std::string& name="")
      :BaseLayer<Dtype>(name),
      inplace_(inplace),
      negative_slope_(negative_slope) {
    }

    virtual void ForwardShape() = 0;
    virtual void ForwardComputation() = 0;

    virtual std::string type() const {
      return "ReLU";
    }

    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<", inplace: "<<(inplace_? "True":"False")
	    <<", Negative_scope: "<<negative_slope_;
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"ReluLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }
    
  protected:
    bool inplace_;

    Dtype negative_slope_;
    
    DISABLE_COPY_AND_ASSIGN(BaseReluLayer);
  }; // class BaseConvLayer
    
} // namespace facethink


#endif
