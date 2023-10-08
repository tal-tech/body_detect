#ifndef __FACETHINK_CAFFE_LAYERS_DETECTION_POSE_OUTPUT_LAYER_CUDA_HPP__
#define __FACETHINK_CAFFE_LAYERS_DETECTION_POSE_OUTPUT_LAYER_CUDA_HPP__


#include "caffe/layers/DetectionPoseOutput/detection_pose_output_layer_base.hpp"

namespace facethink {

  template <typename Dtype>
  class CUDADetectionPoseOutputLayer: public BaseDetectionPoseOutputLayer<Dtype> {
  public:
    explicit CUDADetectionPoseOutputLayer(int num_classes,
					  int num_poses,
					  float nms_threshold, int top_k,
					  float confidence_threshold, int keep_top_k,
					  bool share_location = true,
					  bool share_pose = true,
					  int background_label_id = 0,
					  CodeType code_type = CodeType::CENTER_SIZE,
					  bool variance_encoded_in_target = false,
					  const std::string& name = "")
      :BaseDetectionPoseOutputLayer<Dtype> (num_classes,
					    num_poses,
					    nms_threshold,
					    top_k,
					    confidence_threshold,
					    keep_top_k,
					    share_location,
					    share_pose,
					    background_label_id,
					    code_type,
					    variance_encoded_in_target,
					    name) {
    }

    virtual void ForwardComputation();

    virtual inline std::string type() const {
      return "DetectionPoseOutput(CUDA)";
    }

    DISABLE_COPY_AND_ASSIGN(CUDADetectionPoseOutputLayer);

  }; // class CUDADetectionPoseOutputLayer

} // namespace facethink

#endif
