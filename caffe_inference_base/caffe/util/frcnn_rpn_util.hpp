#ifndef __FACETHINK_CAFFE_FRCNN_RPN_UTIL_HPP__
#define __FACETHINK_CAFFE_FRCNN_RPN_UTIL_HPP__

#include "caffe/core/common.hpp"
#include <opencv2/opencv.hpp>

namespace facethink {
  namespace frcnn {
    
    cv::Mat generate_anchors(const std::vector<float>& scales_v);


    cv::Vec4f bbox_transform_inv(const cv::Vec4f& box,
				 const cv::Vec4f& delta);

    cv::Mat bbox_transform_inv_all(const cv::Mat& box,
				   const cv::Mat& delta);

    cv::Vec4f clip_bbox(const cv::Vec4f& box,
			float im_height, float im_width);
  
    cv::Mat clip_bbox_all(const cv::Mat& box,
			  float height, float width);
  
    float get_iou(const cv::Vec4f &A, const cv::Vec4f &B);



    ///////////////////////////////////////////////////////////////
    void MulStdAndPlusMean(cv::Mat& mat,
			   const std::vector<float>& stds,
			   const std::vector<float>& means);


    void fast_filter(const cv::Mat& bbox_probs, const cv::Mat& pred_bboxes,
		     cv::Mat& fg_idx,  cv::Mat& fg_probs, cv::Mat& fg_bboxes,
		     float conf_thread);


    void fast_nms(const cv::Mat& boxes, const cv::Mat& scores,
		  cv::Mat& idx_nms,
		  float nms_thread);

    void fast_post_processing(const cv::Mat& bbox_probs, const cv::Mat& pred_bboxes,
			      const cv::Mat& pose_probs, const cv::Mat& head_pred_bboxes,
			      cv::Mat& bbox_probs_final, cv::Mat& pred_bboxes_final,
			      cv::Mat& pose_probs_final, cv::Mat& head_pred_bboxes_final,
			      float conf_thread, float nms_thread);

  } // namespace frcnn
} // namespace facethink
#endif
