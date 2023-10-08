#ifndef __FACETHINK_CAFFE_LAYERS_ABSVAL_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_ABSVAL_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {
  
  template <typename Dtype>
  class BaseAbsValLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseAbsValLayer(const std::string& name="")
      :BaseLayer<Dtype>(name) {
    }

    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;

    virtual std::string type() const {
      return "AbsVal";
    }

    virtual inline std::string param_string() const {
      return "";
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"AbsValLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }
      
    DISABLE_COPY_AND_ASSIGN(BaseAbsValLayer);
  }; // class BaseAbsValLayer
    
} // namespace facethink

#endif
