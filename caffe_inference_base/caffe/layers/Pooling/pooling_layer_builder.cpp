#include "caffe/layers/Pooling/pooling_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/Pooling/pooling_layer_cudnn.hpp"
#else
#include "caffe/layers/Pooling/pooling_layer_cpu.hpp"
#endif

namespace facethink {

  
  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  PoolingLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int kernel_size_h, kernel_size_w;
    int pad_h, pad_w;
    int stride_h, stride_w;
    PoolingMode mode;
    bool global_pooling;
    
    ParseLayerParameters(layer_param,
			 kernel_size_h, kernel_size_w,
			 pad_h, pad_w,
			 stride_h, stride_w,
			 mode,
			 global_pooling,
			 name);
    
    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDNNPoolingLayer<Dtype> >(kernel_size_h, kernel_size_w,
							pad_h, pad_w,
							stride_h, stride_w,
							mode,
							global_pooling,
							name);
#else			
    layer = std::make_shared<PoolingLayer<Dtype> >(kernel_size_h, kernel_size_w,
						   pad_h, pad_w,
						   stride_h, stride_w,
						   mode,
						   global_pooling,
						   name);
#endif
    return layer;
  }

  template <typename Dtype>
  void PoolingLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							int& kernel_size_h, int& kernel_size_w,
							int& pad_h, int& pad_w,
							int& stride_h, int& stride_w,
							PoolingMode& mode,
							bool& global_pooling,
							std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::PoolingParameter& pooling_param = layer_param.pooling_param();

    caffe::PoolingParameter_PoolMethod mtd = pooling_param.pool();
    if (mtd == caffe::PoolingParameter_PoolMethod::PoolingParameter_PoolMethod_MAX) {
      mode = POOLING_MAX;
    }else if (mtd ==  caffe::PoolingParameter_PoolMethod::PoolingParameter_PoolMethod_AVE) {
      mode = POOLING_AVE;
    } else {
      BOOST_LOG_TRIVIAL(error)<<"PoolingLayerBuilder: only support max and average pooling.";
    }

    if (pooling_param.has_global_pooling()) {
      global_pooling = pooling_param.global_pooling();
    } else {
      global_pooling = false;
    }

    if (global_pooling) {
      if (pooling_param.has_kernel_size() ||
	  pooling_param.has_kernel_h() ||
	  pooling_param.has_kernel_w()) {
	BOOST_LOG_TRIVIAL(error)<< "PoolingLayerBuilder: With Global_pooling: true Filter size cannot specified.";
      }
    }
    
    if (pooling_param.has_kernel_h() || pooling_param.has_kernel_w()) {
      if (pooling_param.has_kernel_size()) {
	BOOST_LOG_TRIVIAL(error)<< "PoolingLayerBuilder: Either kernel_size or kernel_h/w should be specified; not both.";
      }
      kernel_size_h = pooling_param.kernel_h();
      kernel_size_w = pooling_param.kernel_w();
    } else {
      kernel_size_h = pooling_param.kernel_size();
      kernel_size_w = pooling_param.kernel_size();
    }
    
    if (pooling_param.has_pad_h() || pooling_param.has_pad_w()) {
      if (pooling_param.has_pad()) {
	BOOST_LOG_TRIVIAL(error)<< "PoolingLayerBuilder: Either pad or pad_h/w should be specified; not both.";
      }
      pad_h = pooling_param.pad_h();
      pad_w = pooling_param.pad_w();
    } else if (pooling_param.has_pad()) {
      pad_h = pooling_param.pad();
      pad_w = pooling_param.pad();
    } else {
      pad_h = 0;
      pad_w = 0;
    }

    if (pooling_param.has_stride_h() || pooling_param.has_stride_w()) {
      if (pooling_param.has_stride()) {
	BOOST_LOG_TRIVIAL(error)<< "PoolingLayerBuilder: Either stride or stride_h/w should be specified; not both.";
      }
      stride_h = pooling_param.stride_h();
      stride_w = pooling_param.stride_w();
    } else if(pooling_param.has_stride()) {
      stride_h = pooling_param.stride();
      stride_w = pooling_param.stride();
    } else {
      stride_h = 1;
      stride_w = 1;
    }


    if (global_pooling) {
      if(!(pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1)){
	BOOST_LOG_TRIVIAL(error)<< "PoolingLayerBuilder: With Global_pooling: true; only pad = 0 and stride = 1";
      }
    }
    
  }

} // namespace facethink
