#ifndef __FACETHINK_CAFFE_LAYERS_FLATTEN_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_FLATTEN_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template<typename Dtype>
  class BaseFlattenLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseFlattenLayer(int start_axis = 1,
			      int end_axis = -1,
			      const std::string& name = "")
      :BaseLayer<Dtype>(name),
      start_axis_(start_axis),
      end_axis_(end_axis) {
    }

    virtual void ForwardShape();
    virtual void ForwardComputation() { };

    virtual inline std::string type() const {
      return "Flatten";
    }

    virtual inline std::string param_string() const{
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<", start_axis: "<<start_axis_
	    <<", end_axis: "<<end_axis_;
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"FlattenLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

  protected:
    int start_axis_;
    int end_axis_;

    DISABLE_COPY_AND_ASSIGN(BaseFlattenLayer);
  }; // class BaseFlattenLayer

} // namespace facethink


#endif
