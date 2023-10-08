#include "caffe/layers/Eltwise/eltwise_layer_builder.hpp"


#ifndef CPU_ONLY
#include "caffe/layers/Eltwise/eltwise_layer_cuda.hpp"
#else
#include "caffe/layers/Eltwise/eltwise_layer_cpu.hpp"
#endif


namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  EltwiseLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    EltwiseOp op;
    std::vector<Dtype> coeffs;
    
    ParseLayerParameters(layer_param,
			 op,
			 coeffs,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAEltwiseLayer<Dtype> >(op,
						       coeffs,
						       name);
    
#else
    layer = std::make_shared<EltwiseLayer<Dtype> >(op,
						   coeffs,
						   name);
#endif
    return layer;
  }

  template <typename Dtype>
  void EltwiseLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							EltwiseOp& op,
							std::vector<Dtype>& coeffs,
							std::string& name) {
    this->ParseLayerName(layer_param, name);
    const caffe::EltwiseParameter& elt_param = layer_param.eltwise_param();

    if (elt_param.operation() == caffe::EltwiseParameter_EltwiseOp_PROD) {
      op = EltwiseOp::PROD;
    } else if (elt_param.operation() == caffe::EltwiseParameter_EltwiseOp_SUM) {
      op = EltwiseOp::SUM;
    } else if (elt_param.operation() == caffe::EltwiseParameter_EltwiseOp_MAX) {
      op = EltwiseOp::MAX;
    } else {
      BOOST_LOG_TRIVIAL(error)<<"EltwiseLayerBuilder: Unsupported elementwise operation.";
    }

    coeffs.clear();
    if (elt_param.coeff_size()) {
      for (int i = 0; i < elt_param.coeff_size(); ++i) {
	coeffs.push_back(static_cast<Dtype>(elt_param.coeff(i)));
      }
    }
  }


} // namespace facethink
