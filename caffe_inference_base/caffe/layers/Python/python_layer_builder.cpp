#include "caffe/layers/Python/python_layer_builder.hpp"
#include <boost/algorithm/string.hpp>

#include "caffe/layers/Python/proposal_layer_base.hpp"


namespace facethink {
  
  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  PythonLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    const caffe::PythonParameter& python_param = layer_param.python_param();
    const std::string& model_name = python_param.module();

    if (!model_name.compare("rpn.proposal_layer")) {
      return CreateProposalLayer(layer_param);
    } else {
      BOOST_LOG_TRIVIAL(error)<<"PythonLayerBuilder: Unsupported Python Module.";
      return nullptr;
    }
  }

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  PythonLayerBuilder<Dtype>::CreateProposalLayer(const caffe::LayerParameter& layer_param) {
    std::string name;
    int feat_stride;
    std::vector<Dtype> scales;
    int allowed_border;
    
    ParseProposalLayerParameters(layer_param, feat_stride, scales, allowed_border, name);

    std::shared_ptr<BaseLayer<Dtype> > layer;

    layer = std::make_shared<BaseProposalLayer<Dtype> >(feat_stride,
							scales,
							allowed_border,
							name);
    return layer;
  }

  template <typename Dtype>
  void PythonLayerBuilder<Dtype>::ParseProposalLayerParameters(const caffe::LayerParameter& layer_param,
							       int& feat_stride,
							       std::vector<Dtype>& scales,
							       int& allowed_border,
							       std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::PythonParameter& python_param = layer_param.python_param();
    const std::string& param_str = python_param.param_str();

   
    std::vector<std::string> param_strs;
    boost::split(param_strs,  param_str, boost::is_any_of("\n"));

    for (size_t i=0; i<param_strs.size(); ++i){
      const std::string& p_str = param_strs[i];
      if (p_str.find("feat_stride") != std::string::npos){
	int pos = p_str.find(":");
	std::string sub_str = p_str.substr(pos+1, p_str.length());
	feat_stride = std::stoi(sub_str);
      }else if (p_str.find("scales") != std::string::npos){
	scales.clear();
	int pos_start = p_str.find("[");
	int pos_end = p_str.find("]");
	std::string sub_str = p_str.substr(pos_start+1, pos_end - pos_start -1);

	std::vector<std::string> sub_strs;
	boost::split(sub_strs,  sub_str, boost::is_any_of(","));

	for (size_t k=0; k<sub_strs.size(); k++){
	  scales.push_back(std::stoi(sub_strs[k]));
	}

      }else if (p_str.find("allowed_border") != std::string::npos){
	int pos = p_str.find(":");
	std::string sub_str = p_str.substr(pos+1, p_str.length());
	allowed_border = std::stoi(sub_str);
      }else{
	BOOST_LOG_TRIVIAL(error)<<"ProposalLayer: wrong python param (only support feat_stride, scales, allowed_border).";
      }
    }
    
  }

} // namespace facethink
