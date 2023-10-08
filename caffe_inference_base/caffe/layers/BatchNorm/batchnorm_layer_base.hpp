#ifndef __FACETHINK_CAFFE_LAYERS_BATCH_NORM_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_BATCH_NORM_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseBatchNormLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseBatchNormLayer(bool use_global_stats,
				Dtype moving_average_fraction = Dtype(.999),
				Dtype eps = Dtype(1e-5),
				const std::string& name="")
      :BaseLayer<Dtype>(name),
      use_global_stats_(use_global_stats),
      moving_average_fraction_(moving_average_fraction),
      eps_(eps) {
    }
    
    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;
   
    virtual inline std::string type() const {
      return "BatchNorm";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream <<"("<<this->name_<<")"
	     <<", use_global_stats: "<<(use_global_stats_? "True":"False")
	     <<", moving_average_fraction: "<<moving_average_fraction_
	     <<", eps: "<<eps_;
      return stream.str();
    }

    /// set and output parameters
    virtual inline bool has_weights() const { return true; }
    inline bool use_global_stats() const { return use_global_stats_; }
    inline Dtype moving_average_fraction() const { return moving_average_fraction_; }
    inline Dtype eps() const { return eps_; }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 1 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"BaseBatchNormLayer: only accept one input blob and one output blob.";
	return false;
      }
      return true;
    }

    inline bool CheckWeights() const {
      if (this->weights_.empty()) {
	BOOST_LOG_TRIVIAL(error)<<"BaseBatchNormLayer: Invalid weights or weights have not been imported.";
	return false;
      }
      return true;
    }
    
  protected:
    bool use_global_stats_;
    Dtype moving_average_fraction_;
    Dtype eps_;

    int channels_;
    Blob<Dtype> mean_, variance_, temp_, x_norm_;
    // extra temporarary variables is used to carry out sums/broadcasting
    // using BLAS
    Blob<Dtype> batch_sum_multiplier_;
    Blob<Dtype> num_by_chans_;
    Blob<Dtype> spatial_sum_multiplier_;


    DISABLE_COPY_AND_ASSIGN(BaseBatchNormLayer);
  }; // class BaseBatchNormLayer


} // namespace facethink



#endif
