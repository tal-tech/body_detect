#include "caffe/layers/ROIPooling/roi_pooling_layer_cpu.hpp"
#include <float.h>

namespace facethink {

  template <typename Dtype>
  void ROIPoolForwardCPU(const int index, const Dtype* bottom_data,
			 const Dtype spatial_scale, const int channels, const int height,
			 const int width, const int pooled_height, const int pooled_width,
			 const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
      / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
      / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
					* bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
					* bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
				     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
				     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = std::min(std::max(hstart + roi_start_h, 0), height);
    hend = std::min(std::max(hend + roi_start_h, 0), height);
    wstart = std::min(std::max(wstart + roi_start_w, 0), width);
    wend = std::min(std::max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	int bottom_index = h * width + w;
	if (bottom_data[bottom_index] > maxval) {
	  maxval = bottom_data[bottom_index];
	  maxidx = bottom_index;
	}
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;    
  }
  
  template <typename Dtype>
  void ROIPoolingLayer<Dtype>::ForwardComputation() {
    const Dtype* input_data = this->inputs_[0]->cpu_data();
    const Dtype* input_rois = this->inputs_[1]->cpu_data();
    Dtype* output_data = this->outputs_[0]->mutable_cpu_data();
    int* argmax_data = this->max_idx_.mutable_cpu_data();
    
    int count = this->outputs_[0]->count();

    int channels = this->inputs_[0]->shape(1);
    int height   = this->inputs_[0]->shape(2);
    int width    = this->inputs_[0]->shape(3);

    for (int i = 0; i < count; ++i) {
      ROIPoolForwardCPU<Dtype>(i,
			       input_data,
			       this->spatial_scale_,
			       channels,
			       height, width,
			       this->pooled_h_,
			       this->pooled_w_,
			       input_rois,
			       output_data,
			       argmax_data);
    }
  }
  
  INSTANTIATE_CLASS(ROIPoolingLayer);
} // namespace facethink
