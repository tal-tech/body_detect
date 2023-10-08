#ifndef __FACETHINK_CAFFE_LAYERS_DETECTION_POSE_OUTPUT_LAYER_BASE_HPP__
#define __FACETHINK_CAFFE_LAYERS_DETECTION_POSE_OUTPUT_LAYER_BASE_HPP__

#include "caffe/core/base_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace facethink {

  template <typename Dtype>
  class BaseDetectionPoseOutputLayer: public BaseLayer<Dtype> {
  public:
    explicit BaseDetectionPoseOutputLayer(int num_classes,
					  int num_poses,
					  float nms_threshold, int top_k,
					  float confidence_threshold, int keep_top_k,
					  bool share_location = true,
					  bool share_pose = true,
					  int background_label_id = 0,
					  CodeType code_type = CodeType::CENTER_SIZE,
					  bool variance_encoded_in_target = false,
					  const std::string& name = "")
      :BaseLayer<Dtype>(name),
      num_classes_(num_classes),
      num_poses_(num_poses),
      nms_threshold_(nms_threshold),
      top_k_(top_k),
      confidence_threshold_(confidence_threshold),
      keep_top_k_(keep_top_k),
      share_location_(share_location),
      share_pose_(share_pose),
      background_label_id_(background_label_id),
      code_type_(code_type),
      variance_encoded_in_target_(variance_encoded_in_target) {
    }

    virtual void ForwardShape();
    virtual void ForwardComputation() = 0;

    virtual inline std::string type() const {
      return "DetectionPoseOutput";
    }
   
    virtual inline std::string param_string() const {
      std::ostringstream stream;
      stream<<"("<<this->name_<<")"
	    <<" num_class: "<<num_classes_
	    <<" share_location: "<<(share_location_? "True":"False")
	    <<" num_poses: "<<num_poses_
	    <<" share_pose: "<<(share_pose_? "True":"False")
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
      if (this->inputs_.size() != 4 || this->outputs_.size() != 1) {
	BOOST_LOG_TRIVIAL(error)<<"DetectionPoseOutputLayer: only accept four input blobs and one output blob.";
	return false;
      }
      return true;
    }
    
  protected:
    int num_classes_;
    int num_poses_;

    bool share_location_;
    bool share_pose_;

    int background_label_id_;

    float nms_threshold_;
    int top_k_;
    int keep_top_k_;
    float confidence_threshold_;

    CodeType code_type_;
    bool variance_encoded_in_target_;

    int num_priors_;
    int num_loc_classes_;
    int num_pose_classes_;

    Blob<Dtype> bbox_preds_;
    Blob<Dtype> bbox_permute_;
    Blob<Dtype> conf_permute_;
    Blob<Dtype> pose_permute_;
    
    DISABLE_COPY_AND_ASSIGN(BaseDetectionPoseOutputLayer);
  }; // BaseDetectionPoseOutputLayer

} //namespace facethink



#endif
