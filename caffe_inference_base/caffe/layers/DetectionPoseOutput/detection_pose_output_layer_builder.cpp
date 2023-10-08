#include "caffe/layers/DetectionPoseOutput/detection_pose_output_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/DetectionPoseOutput/detection_pose_output_layer_cuda.hpp"
#else
#include "caffe/layers/DetectionPoseOutput/detection_pose_output_layer_cpu.hpp"
#endif


namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  DetectionPoseOutputLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int num_classes;
    int num_poses;
    float nms_threshold;
    int top_k;
    float confidence_threshold;
    int keep_top_k;
    bool share_location;
    bool share_pose;
    int background_label_id;
    CodeType code_type;
    bool variance_encoded_in_target;
      
    
    ParseLayerParameters(layer_param,
			 num_classes,
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
			 name);
    
    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDADetectionPoseOutputLayer<Dtype> >(num_classes,
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
								   name);
#else			
    layer = std::make_shared<DetectionPoseOutputLayer<Dtype> >(num_classes,
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
							       name);
#endif
    return layer;
  }


  template <typename Dtype>
  void DetectionPoseOutputLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
								    int& num_classes,
								    int& num_poses,
								    float& nms_threshold,
								    int& top_k,
								    float& confidence_threshold,
								    int& keep_top_k,
								    bool& share_location,
								    bool& share_pose,
								    int& background_label_id,
								    CodeType& code_type,
								    bool& variance_encoded_in_target,
								    std::string& name) {
    this->ParseLayerName(layer_param, name);
    
    const caffe::DetectionPoseOutputParameter& do_param = layer_param.detection_pose_output_param();

    if (!do_param.has_num_classes()){
      BOOST_LOG_TRIVIAL(error)<< "DetectionPoseOutLayerBuilder: must specify num_classes";
    }
    num_classes = do_param.num_classes();

    if (do_param.has_share_location()){
      share_location = do_param.share_location();
    }else{
      share_location = true;
    }

    if (!do_param.has_num_poses()){
      BOOST_LOG_TRIVIAL(error)<< "DetectionPoseOutLayerBuilder: must specify num_poses";
    }
    num_poses = do_param.num_poses();

    if (do_param.has_share_pose()){
      share_pose = do_param.share_pose();
    }else{
      share_pose = true;
    }
      
    if (do_param.has_background_label_id()){
      background_label_id = do_param.background_label_id();
    }else{
      background_label_id = 0;
    }

    if (do_param.has_code_type()){
      caffe::PriorBoxParameter_CodeType caffe_code_type = do_param.code_type();
      if (caffe_code_type == caffe::PriorBoxParameter_CodeType::PriorBoxParameter_CodeType_CORNER){
	code_type = CodeType::CORNER;
      }else if (caffe_code_type == caffe::PriorBoxParameter_CodeType::PriorBoxParameter_CodeType_CENTER_SIZE){
	code_type = CodeType::CENTER_SIZE;
      }else{
	BOOST_LOG_TRIVIAL(error)<< "DetectionPoseOutLayerBuilder: unsupported code type.";
      }
    }else{
      code_type = CodeType::CORNER;
    }

    if (do_param.has_variance_encoded_in_target()){
      variance_encoded_in_target = do_param.variance_encoded_in_target();
    }else{
      variance_encoded_in_target = false;
    }

    if (do_param.has_keep_top_k()){
      keep_top_k = do_param.keep_top_k();
    }else{
      keep_top_k = -1;
    }

    if (do_param.has_confidence_threshold()){
      confidence_threshold = do_param.confidence_threshold();
    }else{
      confidence_threshold = 0.0;
    }

    const caffe::NonMaximumSuppressionParameter& nms_param = do_param.nms_param();
    if (nms_param.has_nms_threshold()){
      nms_threshold = nms_param.nms_threshold();
    }else{
      nms_threshold = 0.3;
    }

    if (!nms_param.has_top_k()){
      BOOST_LOG_TRIVIAL(error)<< "DetectionPoseOutLayerBuilder: must specify top_k";
    }
    top_k = nms_param.top_k();
  }

} // namespace facethink

