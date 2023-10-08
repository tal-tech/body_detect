#include "caffe/layers/PriorBox/priorbox_layer_base.hpp"

namespace facethink {
  
  template <typename Dtype>
  void BasePriorBoxLayer<Dtype>::InitLayer() {
    std::vector<Dtype> aspect_ratios_tmp, variances_tmp;
    // aspect_ratios
    aspect_ratios_tmp.push_back(Dtype(1.0));
    for (size_t i = 0; i < aspect_ratios_.size(); ++i) {
      Dtype ar = aspect_ratios_[i];
      bool already_exist = false;
      for (size_t j = 0; j < aspect_ratios_tmp.size(); ++j) {
	if (fabs(ar - aspect_ratios_tmp[j]) < 1e-6){
	  already_exist = true;
	  break;
	}
      }

      if (!already_exist){
	aspect_ratios_tmp.push_back(ar);
	if (flip_){
	  aspect_ratios_tmp.push_back(Dtype(1.0)/ar);
	}
      }
    }

    num_priors_ = aspect_ratios_tmp.size();
    
    // max_size
    if (max_size_ > 0){ ++num_priors_; }

    // variances
    if (!variances_.empty()){
      if (variances_.size() > 1 && variances_.size() != 4){
	BOOST_LOG_TRIVIAL(error)<<"PriorBoxLayer: the size of variances must be 4 (if size > 1).";
      }
      for (size_t i = 0; i < variances_.size(); ++i){
	if (variances_[i] <= 0){
	  BOOST_LOG_TRIVIAL(error)<<"PriorBoxLayer: variances must be positive.";
	}
	variances_tmp.push_back(variances_[i]);
      }
    }else{
      variances_tmp.push_back(Dtype(0.1));
    }
  
 
    aspect_ratios_blob_.ImportFrom(aspect_ratios_tmp);
    variances_blob_.ImportFrom(variances_tmp);
  }
  
  template <typename Dtype>
  void BasePriorBoxLayer<Dtype>::ForwardShape() {
    this->CheckBlobs();
    
    const std::vector<int>& input_shape = this->inputs_[0]->shape();
    
    // Since all images in a batch has same height and width, we only need to
    // generate one set of priors which can be shared across all images.
    
    // 2 channels. First channel stores the mean of each prior coordinate.
    // Second channel stores the variance of each prior coordinate.
    std::vector<int> output_shape = { 1, 2, input_shape[2]*input_shape[3]*num_priors_*4 };
    this->outputs_[0]->Reshape(output_shape);
  }

  INSTANTIATE_CLASS(BasePriorBoxLayer);
 
} // namespace facethink
