#ifndef __FACETHINK_CAFFE_LAYERS_PROPOSAL_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_PROPOSAL_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"
#include <opencv2/opencv.hpp>


namespace facethink {

  template <typename Dtype>
  class BaseProposalLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseProposalLayer(int feat_stride,
			       const std::vector<Dtype>& scales,
			       int allowed_border,
			       const std::string& name)
      :BaseLayer<Dtype>(name),
      feat_stride_(feat_stride),
      scales_(scales),
      allowed_border_(allowed_border) {

      InitLayer();
    }

    void InitLayer();

    virtual void ForwardShape() { this->CheckBlobs(); }

    virtual void ForwardComputation() {
      BOOST_LOG_TRIVIAL(error)<<"ProposalLayer: Must call ForwardComputationWithConfig().";
    }

    virtual void ForwardComputationWithConfig(const RPNConfig<Dtype>& rpn_config);


    virtual inline std::string type() const {
      return "Proposal";
    }

    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<", feat_stride: "<<feat_stride_
	    <<", scales: [";

      for (size_t i=0; i<scales_.size(); ++i){
	stream<<scales_[i]<<",";
      }

      stream<<"] "<<", allowed_border: "<<allowed_border_;
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 3 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"ProposalLayer: only accept three input blobs and one output blob.";
	return false;
      }
      return true;
    }

  protected:
    int feat_stride_;
    std::vector<Dtype> scales_;
    int allowed_border_;

    cv::Mat anchors_;

    DISABLE_COPY_AND_ASSIGN(BaseProposalLayer);
  }; // BaseProposalLayer

} // namespace facethink

#endif
