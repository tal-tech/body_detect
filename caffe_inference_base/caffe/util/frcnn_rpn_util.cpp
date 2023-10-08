#include "caffe/util/frcnn_rpn_util.hpp"

namespace facethink {
  namespace frcnn {
    
    void _whctrs(const cv::Mat& anchor, float& w, float&h, float& x_ctr, float& y_ctr){
      w = anchor.at<float>(0,2) - anchor.at<float>(0,0)  + 1;
      h = anchor.at<float>(0,3) - anchor.at<float>(0,1)  + 1;
      x_ctr = anchor.at<float>(0,0) + 0.5 * (w - 1);
      y_ctr = anchor.at<float>(0,1) + 0.5 * (h - 1);  
    }

    cv::Mat _mkanchors(const cv::Mat& ws, const cv::Mat& hs, float x_ctr, float y_ctr){
      cv::Mat anchor(4, ws.cols, CV_32FC1);

      cv::Mat tmp;
      tmp = x_ctr - 0.5 * (ws - 1);
      tmp.copyTo(anchor.row(0));

      tmp = y_ctr - 0.5 * (hs - 1);
      tmp.copyTo(anchor.row(1));

      tmp = x_ctr + 0.5 * (ws - 1);
      tmp.copyTo(anchor.row(2));
  
      tmp = y_ctr + 0.5 * (hs - 1);
      tmp.copyTo(anchor.row(3));

      return anchor.t();
    }

    cv::Mat _ratio_enum(const cv::Mat& anchor, const cv::Mat& ratios){
      float w, h, x_ctr, y_ctr;
      _whctrs(anchor, w, h, x_ctr, y_ctr);

      float size = w*h;

      cv::Mat ws(ratios.size(), CV_32FC1);
      cv::Mat hs(ratios.size(), CV_32FC1);

      for (int i=0; i<ratios.cols; ++i){
	ws.at<float>(0, i) = cvRound(sqrt(size/ratios.at<float>(0, i)));
	hs.at<float>(0, i) = cvRound(ws.at<float>(0, i)*ratios.at<float>(0, i));
      }

 
      cv::Mat anchors = _mkanchors(ws, hs, x_ctr, y_ctr);
      return anchors;
    }

    cv::Mat _scale_enum(const cv::Mat& anchor, const cv::Mat& scales){
      float w, h, x_ctr, y_ctr;
      _whctrs(anchor, w, h, x_ctr, y_ctr);

      cv::Mat ws = w * scales;
      cv::Mat hs = h * scales;

      cv::Mat anchors = _mkanchors(ws, hs, x_ctr, y_ctr);
      return anchors;
    }
  
    cv::Mat generate_anchors(const std::vector<float>& scales_v){
      //  Generate anchor (reference) windows by enumerating aspect ratios X
      // scales wrt a reference (0, 0, 15, 15) window.
      int base_size = 16;
      cv::Mat ratios = (cv::Mat_<float>(1,3) << 0.5, 1, 2);
      cv::Mat base_anchor = (cv::Mat_<float>(1,4) << 0, 0, base_size-1, base_size-1);

      cv::Mat scales = cv::Mat(scales_v, false).reshape(1, 1);

      //Enumerate a set of anchors for each aspect ratio wrt an anchor.
    
      cv::Mat ratio_anchors = _ratio_enum(base_anchor, ratios);
      cv::Mat anchors(ratio_anchors.rows * scales.cols, 4, CV_32FC1);
    
      for (size_t i=0; i<ratio_anchors.rows; ++i){
	cv::Mat tmp = _scale_enum(ratio_anchors.row(i), scales);
	tmp.copyTo(anchors(cv::Range(i*scales.cols, (i+1)*scales.cols), cv::Range::all()));
      }
  
      return anchors;
    }


    /////////////////////////////////////////////////////////////////////////////////////
    cv::Vec4f bbox_transform_inv(const cv::Vec4f& box,
				 const cv::Vec4f& delta){
      float src_w = box[2] - box[0] + 1;
      float src_h = box[3] - box[1] + 1;
      float src_ctr_x = box[0] + 0.5 * src_w; // box[0] + 0.5*src_w;
      float src_ctr_y = box[1] + 0.5 * src_h; // box[1] + 0.5*src_h;
      float pred_ctr_x = delta[0] * src_w + src_ctr_x;
      float pred_ctr_y = delta[1] * src_h + src_ctr_y;
      float pred_w = exp(delta[2]) * src_w;
      float pred_h = exp(delta[3]) * src_h;

      return cv::Vec4f(pred_ctr_x - 0.5 * pred_w,
		       pred_ctr_y - 0.5 * pred_h,
		       pred_ctr_x + 0.5 * pred_w,
		       pred_ctr_y + 0.5 * pred_h);
    }
    
    ///////////////////////////////////////////////////////////////
    cv::Mat bbox_transform_inv(const cv::Mat& box,
			       const cv::Mat& delta){
      return cv::Mat(bbox_transform_inv(cv::Vec4f(box),
					cv::Vec4f(delta))).t();
    }

    cv::Mat bbox_transform_inv_all(const cv::Mat& box,
				   const cv::Mat& delta){

      cv::Mat out(delta.size(), CV_32FC1);
    
      for (int i=0; i<delta.rows; ++i){
	for (int j=0; j<delta.cols/4; ++j){
	  cv::Mat tmp = bbox_transform_inv(box.row(i), delta(cv::Range(i,i+1), cv::Range(j*4, (j+1)*4)));
	  tmp.copyTo(out(cv::Range(i,i+1), cv::Range(j*4, (j+1)*4)));
	}
      }
    
      return out;
    }

    cv::Vec4f clip_bbox(const cv::Vec4f& box,
			float im_height,
			float im_width){
      cv::Vec4f out;
      out[0] = std::max(0.0f, std::min(box[0], im_width  - 1));
      out[1] = std::max(0.0f, std::min(box[1], im_height - 1));
      out[2] = std::max(0.0f, std::min(box[2], im_width  - 1));
      out[3] = std::max(0.0f, std::min(box[3], im_height - 1));
      return out;
    }

    cv::Mat clip_bbox(const cv::Mat& box,
		      float im_height,
		      float im_width){
      return cv::Mat(clip_bbox(cv::Vec4f(box), im_height, im_width)).t();
    }
  
    cv::Mat clip_bbox_all(const cv::Mat& box,
			  float im_height,
			  float im_width){
      cv::Mat out(box.size(), CV_32FC1);
    
      for (int i=0; i<box.rows; ++i){
	for (int j=0; j<box.cols/4; ++j){
	  cv::Mat tmp = clip_bbox(box(cv::Range(i,i+1), cv::Range(j*4, (j+1)*4)),
				  im_height,
				  im_width);
	  tmp.copyTo(out(cv::Range(i,i+1), cv::Range(j*4, (j+1)*4)));
	}
      }
    
      return out;
    }

    float get_iou(const cv::Vec4f &A, const cv::Vec4f &B) {
      const float xx1 = std::max(A[0], B[0]);
      const float yy1 = std::max(A[1], B[1]);
      const float xx2 = std::min(A[2], B[2]);
      const float yy2 = std::min(A[3], B[3]);
      float inter = std::max(float(0), xx2 - xx1 + 1) * std::max(float(0), yy2 - yy1 + 1);
      float areaA = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
      float areaB = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
      return inter / (areaA + areaB - inter);
    }

    float get_iou(const cv::Mat& A, const cv::Mat& B){
      return get_iou(cv::Vec4f(A),  cv::Vec4f(B));
    }

  
    /////////////////////////////////////////////////////////////////////////////

    void MulStdAndPlusMean(cv::Mat& mat, const std::vector<float>& stds, const std::vector<float>& means){
      CV_Assert(mat.cols == stds.size() && stds.size() == means.size());

      for (int i=0; i<mat.rows; ++i){
	for (int j=0; j<mat.cols; ++j){
	  mat.at<float>(i,j) = mat.at<float>(i,j) * stds[j] + means[j];
	}
      }
    }


    void fast_filter(const cv::Mat& bbox_probs, const cv::Mat& pred_bboxes,
		     cv::Mat& fg_idx,  cv::Mat& fg_probs, cv::Mat& fg_bboxes,
		     float conf_thread){
      int max_idx;
      float max_val;

      std::vector<int> row_selected, col_selected;
      for (int i=0; i<bbox_probs.rows; ++i){
	max_idx = 0;
	max_val = bbox_probs.at<float>(i,0);;
	for (int j=1; j<bbox_probs.cols; ++j){
	  if (max_val < bbox_probs.at<float>(i,j)){
	    max_val = bbox_probs.at<float>(i,j);
	    max_idx = j;
	  }
	}

	if (max_idx > 0 && max_val > conf_thread){
	  row_selected.push_back(i);
	  col_selected.push_back(max_idx);
	}
      }

      int n_selected = row_selected.size(); 
      fg_idx = cv::Mat(n_selected, 1, CV_32SC1);
      fg_probs = cv::Mat(n_selected, 1, CV_32FC1);
      fg_bboxes = cv::Mat(n_selected, 4, CV_32FC1);

      for (size_t i=0; i<row_selected.size(); ++i){
	fg_idx.at<int>(i, 0) = row_selected[i];
	fg_probs.at<float>(i,0) = bbox_probs.at<float>(row_selected[i], col_selected[i]);
	pred_bboxes(cv::Range(row_selected[i],row_selected[i]+1),
		    cv::Range(col_selected[i]*4, (col_selected[i]+1)*4)).copyTo(fg_bboxes.row(i));
      }
    }



    void fast_nms(const cv::Mat& boxes, const cv::Mat& scores,
		  cv::Mat& idx_nms,
		  float nms_thread){
      typedef std::pair<float, int> sort_pair;
      std::vector<sort_pair> sort_vector;

      for (size_t i=0; i<scores.rows; ++i){
	sort_vector.push_back(sort_pair(scores.at<float>(i,0), i));
      }
      std::sort(sort_vector.begin(), sort_vector.end(), std::greater<sort_pair>());

      const int n_box = sort_vector.size();
      std::vector<bool> select(n_box, true);

      std::vector<int> idx_select;
      for (int i=0; i<n_box; ++i){
	if (select[i]){
	  const int cur_i = sort_vector[i].second;
	  for (int j=i+1; j<n_box; ++j){
	    if (select[j]){
	      const int cur_j = sort_vector[j].second;
	      if (get_iou(boxes.row(cur_i), boxes.row(cur_j)) > nms_thread){
		select[j] = false;
	      }
	    }
	  }
	  idx_select.push_back(cur_i);
	}
      }

      int n_selected = idx_select.size();
      idx_nms = cv::Mat(n_selected, 1, CV_32SC1);

      for (int i=0; i<n_selected; ++i){
	idx_nms.at<int>(i,0) = idx_select[i];
      }
    }


    void fast_post_processing(const cv::Mat& bbox_probs, const cv::Mat& pred_bboxes,
			      const cv::Mat& pose_probs, const cv::Mat& head_pred_bboxes,
			      cv::Mat& bbox_probs_final, cv::Mat& pred_bboxes_final,
			      cv::Mat& pose_probs_final, cv::Mat& head_pred_bboxes_final,
			      float conf_thread, float nms_thread){

      cv::Mat fg_idx, fg_probs, fg_bboxes;
      fast_filter(bbox_probs, pred_bboxes, fg_idx, fg_probs, fg_bboxes, conf_thread);

      cv::Mat idx_nms;
      fast_nms(fg_bboxes, fg_probs, idx_nms, nms_thread);

      int num_keep = idx_nms.rows;
    
      bbox_probs_final = cv::Mat(num_keep, bbox_probs.cols, bbox_probs.type());
      pred_bboxes_final = cv::Mat(num_keep, pred_bboxes.cols, pred_bboxes.type());
      pose_probs_final = cv::Mat(num_keep, pose_probs.cols, pose_probs.type());
      head_pred_bboxes_final = cv::Mat(num_keep, head_pred_bboxes.cols, head_pred_bboxes.type());
    
      for (int i=0; i<num_keep; ++i){
	int idx_keep = fg_idx.at<int>(idx_nms.at<int>(i,0), 0);

	bbox_probs.row(idx_keep).copyTo(bbox_probs_final.row(i));
	pred_bboxes.row(idx_keep).copyTo(pred_bboxes_final.row(i));
	pose_probs.row(idx_keep).copyTo(pose_probs_final.row(i));
	head_pred_bboxes.row(idx_keep).copyTo(head_pred_bboxes_final.row(i));
      }
    }

    
  } // namespace frcnn
} // namespace facethink
