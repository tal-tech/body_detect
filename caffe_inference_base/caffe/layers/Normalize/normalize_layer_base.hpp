#ifndef __FACETHINK_CAFFE_LAYERS_NORMALIZE_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_NORMALIZE_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseNormalizeLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseNormalizeLayer(bool across_spatial = true,
				bool channel_shared = true,
				Dtype eps = Dtype(1e-10),
				const std::string& name="")
      :BaseLayer<Dtype>(name),
      across_spatial_(across_spatial),
      channel_shared_(channel_shared),
      eps_(eps) {
    }
    
    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;
   
    virtual inline std::string type() const {
      return "Normalize";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream <<"("<<this->name_<<")"
	     <<", across_spatial: "<<(across_spatial_? "True":"False")
	     <<", channel_shared: "<<(channel_shared_? "True":"False")
	     <<", eps: "<<eps_;
      return stream.str();
    }

    /// set and output parameters
    virtual inline bool has_weights() const { return true; }
    inline bool across_spatial() const { return across_spatial_; }
    inline bool channel_shared() const { return channel_shared_; }
    inline Dtype eps() const { return eps_; }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"NormalizeLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

    inline bool CheckWeights() const {
      if (this->weights_.empty()) {
	BOOST_LOG_TRIVIAL(error)<<"NormalizeLayer: Invalid weights or weights have not been imported.";
	return false;
      }
      return true;
    }
    
  protected:
    bool across_spatial_;
    bool channel_shared_;

    Dtype eps_;

    Blob<Dtype> norm_;
    Blob<Dtype> buffer_;
    Blob<Dtype> sum_channel_multiplier_, sum_spatial_multiplier_;


    DISABLE_COPY_AND_ASSIGN(BaseNormalizeLayer);
  }; // class BaseNormalizeLayer


} // namespace facethink



#endif
