#ifndef __FACETHINK_CAFFE_LAYERS_TANH_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_TANH_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseTanHLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseTanHLayer( const std::string& name="" )
      :BaseLayer<Dtype>(name) {
    }

    virtual void ForwardShape() = 0;
    virtual void ForwardComputation() = 0;

    virtual std::string type() const {
      return "TanH";
    }

    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")";
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"TanHLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }
    
  protected:
    
    DISABLE_COPY_AND_ASSIGN(BaseTanHLayer);
  }; // class BaseTanHLayer

} // namespace facethink

#endif
