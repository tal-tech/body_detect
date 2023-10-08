#ifndef __FACETHINK_CAFFE_LAYERS_DETECTION_OUTPUT_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_DETECTION_OUTPUT_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace facethink {
  
  template <typename Dtype>
  class BaseDetectionOutputLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseDetectionOutputLayer(int num_classes,
				      float nms_threshold, int top_k, 
				      float confidence_threshold, int keep_top_k,
				      bool share_location = true,
				      int background_label_id = 0,
				      CodeType code_type = CodeType::CENTER_SIZE,
				      bool variance_encoded_in_target = false,
				      const std::string& name = "")
      :BaseLayer<Dtype>(name),
      num_classes_(num_classes),
      nms_threshold_(nms_threshold),
      top_k_(top_k),
      confidence_threshold_(confidence_threshold),
      keep_top_k_(keep_top_k),
      share_location_(share_location),
      background_label_id_(background_label_id),
      code_type_(code_type),
      variance_encoded_in_target_(variance_encoded_in_target) {

    }
    
     
    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;

    virtual inline std::string type() const {
      return "DetectionOutput";
    }
    
    inline std::string param_string() const{
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<" num_class: "<<num_classes_
	    <<" share_location: "<<(share_location_? "True":"False")
	    <<" background_label_id: "<<background_label_id_
	    <<" nms_threshold: "<<nms_threshold_
	    <<" top_k: "<<top_k_;
      
      if (code_type_ == CodeType::CORNER){
	stream<<" code_type: CORNER";
      }else if (code_type_ == CodeType::CENTER_SIZE){
	stream<<" code_type: CENTER_SIZE";
      }
      stream<<" variance_encoded_in_target: "<<(variance_encoded_in_target_? "True":"False")
	    <<" keep_top_k: "<<keep_top_k_
	    <<" confidence_threshold: "<<confidence_threshold_;
      return stream.str();
    }

  protected:
    virtual inline bool CheckBlobs() const {
      if (this->inputs_.size() != 3 || !(this->outputs_.size() == 1 || this->outputs_.size() == 2)) {
	BOOST_LOG_TRIVIAL(error)<<"DetectionOutputLayer: only accept three input blobs and one or two output blobs.";
	return false;
      }
      return true;
    }
    
  protected:
    int num_classes_;
    
    bool share_location_;
    
    int background_label_id_;
    
    float nms_threshold_;
    int top_k_;
    int keep_top_k_;
    float confidence_threshold_;
    
    CodeType code_type_;
    bool variance_encoded_in_target_;
    
    int num_priors_;
    int num_loc_classes_;
    
    Blob<Dtype> bbox_preds_;
    Blob<Dtype> bbox_permute_;
    Blob<Dtype> conf_permute_;
    
    DISABLE_COPY_AND_ASSIGN(BaseDetectionOutputLayer);
  };


}


#endif
