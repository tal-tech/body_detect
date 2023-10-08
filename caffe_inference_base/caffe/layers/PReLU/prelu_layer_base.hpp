#ifndef __FACETHINK_CAFFE_LAYERS_PRELU_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_PRELU_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BasePReluLayer: public BaseLayer<Dtype> {
  public:
    explicit BasePReluLayer(bool inplace = true, bool channel_shared = true, const std::string& name="")
      :BaseLayer<Dtype>(name), inplace_(inplace), channel_shared_(channel_shared) {
    }

    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;

    virtual std::string type() const {
      return "PReLU";
    }

	virtual inline std::string param_string() const {
		std::ostringstream stream;
		stream << "(" << this->name_ << ")"
			<< ", inplace: " << (inplace_ ? "True" : "False")
			<< ", channel_shared: " << (channel_shared_ ? "True" : "False");
		return stream.str();
	}

	inline int channel_shared() const { return channel_shared_; }
	inline int inplace() const { return inplace_; }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"PReluLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

	inline bool CheckWeights() const {
		if (this->weights_.empty()) {
			BOOST_LOG_TRIVIAL(error) << "BasePReluLayer: Invalid weights or weights have not been imported.";
			return false;
		}
		return true;
	}

  protected:
    bool channel_shared_;
	bool inplace_;

    DISABLE_COPY_AND_ASSIGN(BasePReluLayer);
  }; // class BasePReluLayer

} // namespace facethink


#endif
