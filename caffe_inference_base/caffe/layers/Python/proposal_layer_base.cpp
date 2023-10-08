#include "caffe/layers/Python/proposal_layer_base.hpp"
#include "caffe/util/frcnn_rpn_util.hpp"

namespace facethink {

  template <typename Dtype>
  void BaseProposalLayer<Dtype>::InitLayer() {
    anchors_ = frcnn::generate_anchors(scales_);
  }

  template <typename Dtype>
  void BaseProposalLayer<Dtype>::ForwardComputationWithConfig(const RPNConfig<Dtype>& rpn_config) {
    const Dtype *bottom_rpn_score = this->inputs_[0]->cpu_data();  // rpn_cls_prob_reshape
    const Dtype *bottom_rpn_bbox  = this->inputs_[1]->cpu_data();   // rpn_bbox_pred
    const Dtype *bottom_im_info   = this->inputs_[2]->cpu_data();    // im_info

    const std::vector<int> input_shape_1 = this->inputs_[1]->shape();
    const int num      = input_shape_1[0];
    const int channels = input_shape_1[1];
    const int height   = input_shape_1[2];
    const int width    = input_shape_1[3];

    if (num != 1){
      BOOST_LOG_TRIVIAL(error)<<"ProposalLayer: only single item batches are supported.";
      return;
    }
    if (channels % 4 != 0){
      BOOST_LOG_TRIVIAL(error)<<"ProposalLayer: rpn bbox pred channels should be divided by 4";
      return;
    }

    ////////////////////    
    const Dtype im_height = bottom_im_info[0];
    const Dtype im_width  = bottom_im_info[1];
    //std::cout<<"im_size: "<<im_height<<"  "<<im_width<<std::endl;
    //  const float bounds[4] = { im_width - 1, im_height - 1, im_width - 1, im_height -1 };
    const Dtype min_size = bottom_im_info[2] * rpn_config.RPN_MIN_SIZE;


    std::vector<cv::Vec4f> proposals;
    typedef std::pair<float, int> sort_pair;
    std::vector<sort_pair> sort_vector;

    int config_n_anchors = anchors_.rows;
    const float* anchors_p = anchors_.ptr<float>(0);
    cv::Vec4f box, delta;
    for (int j = 0; j < height; j++){
      for (int i = 0; i < width; i++){
	for (int k = 0; k < config_n_anchors; k++){
	  float score = bottom_rpn_score[config_n_anchors * height * width +
					 k * height * width + j * width + i];
	  
	  box[0] = anchors_p[k * 4 + 0] + i * feat_stride_;
	  box[1] = anchors_p[k * 4 + 1] + j * feat_stride_;
	  box[2] = anchors_p[k * 4 + 2] + i * feat_stride_;
	  box[3] = anchors_p[k * 4 + 3] + j * feat_stride_;
	  
	  delta[0] = bottom_rpn_bbox[(k * 4 + 0) * height * width + j * width + i];
	  delta[1] = bottom_rpn_bbox[(k * 4 + 1) * height * width + j * width + i];
	  delta[2] = bottom_rpn_bbox[(k * 4 + 2) * height * width + j * width + i];
	  delta[3] = bottom_rpn_bbox[(k * 4 + 3) * height * width + j * width + i];

	  // Convert anchors into proposals via bbox transformations
	  cv::Vec4f cbox = frcnn::bbox_transform_inv(box, delta);

	  // 2. clip predicted boxes to image
	  cbox = frcnn::clip_bbox(cbox, im_height, im_width);
	  // 3. remove predicted boxes with either height or width < threshold
	  if((cbox[2] - cbox[0] + 1) >= min_size && (cbox[3] - cbox[1] + 1) >= min_size) {
	    const int now_index = sort_vector.size();
	    sort_vector.push_back(sort_pair(score, now_index)); 
	    proposals.push_back(cbox);
	  }
	}
      }
    }

    //  std::cout<<"proposals size: "<<proposals.size()<<std::endl;

    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000)
    std::sort(sort_vector.begin(), sort_vector.end(), std::greater<sort_pair>());
    const int n_proposals = std::min((int)sort_vector.size(), rpn_config.RPN_PRE_NMS_TOP_N);
    sort_vector.erase(sort_vector.begin() + n_proposals, sort_vector.end());
    std::vector<bool> select(n_proposals, true);

    // std::cout<<"select size: "<<n_proposals<<std::endl;

    // apply nms
    std::vector<cv::Vec4f> box_final;
    std::vector<float> scores_final;
    for (int i = 0; i < n_proposals && box_final.size() < rpn_config.RPN_POST_NMS_TOP_N; i++){
      if (select[i]){
	const int cur_i = sort_vector[i].second;
	for (int j = i + 1; j < n_proposals; j++){
	  if (select[j]){
	    const int cur_j = sort_vector[j].second;
	    if (frcnn::get_iou(proposals[cur_i], proposals[cur_j]) > rpn_config.RPN_NMS_THRESH){
	      select[j] = false;
	    }
	  }
	}
	box_final.push_back(proposals[cur_i]);
	scores_final.push_back(sort_vector[i].first);
      }
    }

    // Output rois blob
    // Our RPN implementation only supports a single input image, so all
    // batch inds are 0
    this->outputs_[0]->Reshape(std::vector<int> { (int)box_final.size(), 5});
    Dtype * output_data = this->outputs_[0]->mutable_cpu_data();
    for (size_t i=0; i<box_final.size(); ++i){
      output_data[i*5] = Dtype(0);
      output_data[i*5 + 1] = static_cast<Dtype>(box_final[i][0]);
      output_data[i*5 + 2] = static_cast<Dtype>(box_final[i][1]);
      output_data[i*5 + 3] = static_cast<Dtype>(box_final[i][2]);
      output_data[i*5 + 4] = static_cast<Dtype>(box_final[i][3]);
    }

    if (this->outputs_.size() > 1){
      this->outputs_[1]->Reshape(std::vector<int> { (int)scores_final.size(), 1});
      Dtype * output_score = this->outputs_[0]->mutable_cpu_data();
      for (size_t i=0; i<scores_final.size(); ++i){
	output_score[i] = scores_final[i];
      }
    }
  }

  INSTANTIATE_CLASS(BaseProposalLayer);
} // namespace facethink
