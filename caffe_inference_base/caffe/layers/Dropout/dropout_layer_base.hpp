#ifndef __FACETHINK_CAFFE_LAYERS_DROPOUT_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_DROPOUT_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseDropoutLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseDropoutLayer(Dtype dropout_ratio,
			      const std::string& name="")
      :BaseLayer<Dtype>(name),
      dropout_ratio_(dropout_ratio) {
    }
    
    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;
   
    virtual inline std::string type() const {
      return "Dropout";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<" dropout_ratio: "<<dropout_ratio_;
      return stream.str();
    }


  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"DropoutLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

  protected:
    Dtype dropout_ratio_;

    DISABLE_COPY_AND_ASSIGN(BaseDropoutLayer);
  }; // class BaseDropoutLayer

} // namespace facethink



#endif
