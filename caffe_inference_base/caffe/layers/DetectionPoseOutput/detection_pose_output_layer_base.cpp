#include "caffe/layers/DetectionPoseOutput/detection_pose_output_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseDetectionPoseOutputLayer<Dtype>::ForwardShape() {
    // input :
    // 0. loc 1. conf, 2. pose, 3. priorbox
    this->CheckBlobs();

    num_priors_ = this->inputs_[3]->shape(2) / 4;
    num_loc_classes_ = share_location_? 1 : num_classes_;
    num_pose_classes_ = share_pose_? 1: num_classes_;

    if (num_priors_ * num_loc_classes_ * 4 != this->inputs_[0]->shape(1)){
      BOOST_LOG_TRIVIAL(error)<<"DetectionPoseOutLayer: Number of priors must match number of location predictions.";
    }
    if (num_priors_ * num_classes_ != this->inputs_[1]->shape(1)){
      BOOST_LOG_TRIVIAL(error)<<"DetectionPoseOutLayer: Number of priors must match number of confidence predictions.";
    }
    if (num_priors_ * num_pose_classes_ * num_poses_ != this->inputs_[2]->shape(1)){
      BOOST_LOG_TRIVIAL(error)<<"DetectionPoseOutLayer: Number of priors must match number of pose predictions.";
    }       


    bbox_preds_.Reshape(this->inputs_[0]->shape());
    if (!share_location_){
      bbox_permute_.Reshape(this->inputs_[0]->shape());
    }
    conf_permute_.Reshape(this->inputs_[1]->shape());
    
    pose_permute_.Reshape(this->inputs_[2]->shape());


    // Each row is a 9 dimension vector, which stores
    // [image_id, label, confidence, xmin, ymin, xmax, ymax, pose, pose_confidence]
    this->outputs_[0]->Reshape( std::vector<int> { 1, 9 } );

  }

  INSTANTIATE_CLASS(BaseDetectionPoseOutputLayer);
} // namespace facethink
