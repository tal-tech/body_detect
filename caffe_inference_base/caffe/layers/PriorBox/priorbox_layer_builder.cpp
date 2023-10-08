#include "caffe/layers/PriorBox/priorbox_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/PriorBox/priorbox_layer_cuda.hpp"
#else
#include "caffe/layers/PriorBox/priorbox_layer_cpu.hpp"
#endif

namespace facethink {
  
  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  PriorBoxLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    Dtype min_size, max_size;
    std::vector<Dtype> aspect_ratios, variances;
    bool flip, clip;
    int img_h, img_w;
    Dtype step_h, step_w;
    Dtype offset;
    
    ParseLayerParameters(layer_param,
			 min_size, max_size,
			 aspect_ratios,
			 variances,
			 flip,
			 clip,
			 img_h, img_w,
			 step_h, step_w,
			 offset,
			 name);
    
    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAPriorBoxLayer<Dtype> >(min_size,
							max_size,
							aspect_ratios,
							variances,
							flip,
							clip,
							img_h, img_w,
							step_h, step_w,
							offset,
							name);    
#else			
    layer = std::make_shared<PriorBoxLayer<Dtype> >(min_size,
						    max_size,
						    aspect_ratios,
						    variances,
						    flip,
						    clip,
						    img_h, img_w,
						    step_h, step_w,
						    offset,
						    name);    
#endif
    return layer;
  }

  template <typename Dtype>
  void PriorBoxLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							 Dtype& min_size,
							 Dtype& max_size,
							 std::vector<Dtype>& aspect_ratios,
							 std::vector<Dtype>& variances,
							 bool& flip,
							 bool& clip,
							 int& img_h, int& img_w,
							 Dtype& step_h, Dtype& step_w,
							 Dtype& offset,
							 std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::PriorBoxParameter& prior_box_param = layer_param.prior_box_param();

    if (prior_box_param.has_min_size()){
      min_size = static_cast<Dtype>(prior_box_param.min_size());
    }else{
      BOOST_LOG_TRIVIAL(error)<<"PriorBoxLayerBuilder: must provide min_size.";
    }

    if (prior_box_param.has_max_size()){
      max_size = static_cast<Dtype>(prior_box_param.max_size());
      if (max_size <= min_size){
	BOOST_LOG_TRIVIAL(error)<<"PriorBoxLayerBuilder: max_size must be greater than min_size.";
      }
    }else{
      max_size = Dtype(-1.0);
    }

    aspect_ratios.clear();
    for (int i=0; i<prior_box_param.aspect_ratio_size(); ++i){
      aspect_ratios.push_back(prior_box_param.aspect_ratio(i));
    }

    variances.clear();
    for (int i=0; i < prior_box_param.variance_size(); ++i) {
      variances.push_back(prior_box_param.variance(i));
    }
    
    if (prior_box_param.has_flip()){
      flip = prior_box_param.flip();
    }else{
      flip = true;
    }

    if (prior_box_param.has_clip()){
      clip = prior_box_param.clip();
    }else{
      clip = false;
    }

    
    if (prior_box_param.has_img_h() || prior_box_param.has_img_w()){
      if (prior_box_param.has_img_size())
        BOOST_LOG_TRIVIAL(error)<< "PriorBoxLayer: Either img_size or img_h/img_w should be specified; not both.";
      img_h = prior_box_param.img_h();
      if (img_h <= 0)
	BOOST_LOG_TRIVIAL(error)<< "PriorBoxLayer: img_h should be larger than 0.";
      img_w = prior_box_param.img_w();
      if (img_w <= 0)
	BOOST_LOG_TRIVIAL(error)<< "PriorBoxLayer: img_w should be larger than 0.";
    }else if(prior_box_param.has_img_size()){
      const int img_size = prior_box_param.img_size();
      if (img_size <= 0)
	BOOST_LOG_TRIVIAL(error)<< "PriorBoxLayer: img_size should be larger than 0.";
      img_h = img_size;
      img_w = img_size;
    }else{
      img_h = 0;
      img_w = 0;
    }
    
    if (prior_box_param.has_step_h() || prior_box_param.has_step_w()) {
      if(prior_box_param.has_step())
        BOOST_LOG_TRIVIAL(error)<< "PriorBoxLayer: Either step or step_h/step_w should be specified; not both.";
      step_h = prior_box_param.step_h();
      if (step_h <= 0.)
	BOOST_LOG_TRIVIAL(error)<< "PriorBoxLayer: step_h should be larger than 0.";
      step_w = prior_box_param.step_w();
      if (step_w <= 0.)
	BOOST_LOG_TRIVIAL(error)<< "PriorBoxLayer: step_w should be larger than 0.";
    }else if (prior_box_param.has_step()){
      const float step = prior_box_param.step();
      if (step <= 0)
	BOOST_LOG_TRIVIAL(error)<< "PriorBoxLayer: step should be larger than 0.";
      step_h = step;
      step_w = step;
    }else{
      step_h = 0.;
      step_w = 0.;
    }

    if (prior_box_param.has_offset()){
      offset = prior_box_param.offset();
    }else{
      offset = Dtype(0.5);
    }
  }

} // namespace facethink


