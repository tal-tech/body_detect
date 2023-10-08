#include "caffe/layers/Convolution/conv_layer_builder.hpp"
#include "caffe/util/blob_util.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/Convolution/conv_layer_cudnn.hpp"
#include "caffe/layers/Convolution/conv_layer_cuda.hpp"
#else
#include "caffe/layers/Convolution/conv_layer_cpu.hpp"
#endif


namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  ConvLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int num_output;
    bool bias_term;
    int kernel_size_h, kernel_size_w;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    int group;
    
    ParseLayerParameters(layer_param,
			 num_output,
			 kernel_size_h, kernel_size_w,
			 pad_h, pad_w,
			 stride_h, stride_w,
			 dilation_h, dilation_w,
			 bias_term, group, name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDNNConvLayer<Dtype> >(num_output,
						     kernel_size_h, kernel_size_w,
						     pad_h, pad_w,
						     stride_h, stride_w,
						     dilation_h, dilation_w,
						     bias_term, group, name);
   
#else
    layer = std::make_shared<ConvLayer<Dtype> >(num_output,
						kernel_size_h, kernel_size_w,
						pad_h, pad_w,
						stride_h, stride_w,
						dilation_h, dilation_w,
						bias_term, group, name);
   
#endif
    return layer;
  }


  template<typename Dtype>
  void ConvLayerBuilder<Dtype>::ImportData(const caffe::LayerParameter& layer_param,
					   std::shared_ptr<BaseLayer<Dtype> >& layer) {
    std::shared_ptr<BaseConvLayer<Dtype> > conv_layer =
      std::static_pointer_cast<BaseConvLayer<Dtype> >(layer);
    
    if (IsShapeEqual(layer_param, conv_layer)) {
      std::vector<std::shared_ptr<Blob<Dtype> > > weights;
      std::shared_ptr<Blob<Dtype> > filter_weight = std::make_shared<Blob<Dtype> >();
      BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(0), filter_weight);
      weights.push_back(filter_weight);
      
      if (conv_layer->has_bias_term()) {
	std::shared_ptr<Blob<Dtype> > bias_weight = std::make_shared<Blob<Dtype> >();
	BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(1), bias_weight);
	weights.push_back(bias_weight);
      }

      conv_layer->SetWeights(weights);
      
      BOOST_LOG_TRIVIAL(info)<<"ConvLayerBuilder: Layer("<<conv_layer->name()<<") import" 
			     <<" Weights: "<<weights[0]->shape_string()
			     <<", Bias: "<< (conv_layer->has_bias_term()? weights[1]->shape_string():"NULL");
    } else {
      BOOST_LOG_TRIVIAL(error)<< "ConvLayerBuilder: LayerParameter not match.";
    }
  }

  template <typename Dtype>
  bool ConvLayerBuilder<Dtype>::IsShapeEqual(const caffe::LayerParameter& layer_param,
					     std::shared_ptr<BaseConvLayer<Dtype> >& layer) {
    std::string name;
    bool bias_term;
    int num_output;

    int kernel_size_h, kernel_size_w;
    int pad_h, pad_w; 
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    int group;

    ParseLayerParameters(layer_param,
			 num_output,
			 kernel_size_h, kernel_size_w,
			 pad_h, pad_w,
			 stride_h, stride_w,
			 dilation_h, dilation_w,
			 bias_term, group, name);

    return (layer->name()          == name          &&
	    layer->has_bias_term() == bias_term     &&
	    layer->num_output()    == num_output    &&
	    layer->kernel_size_h() == kernel_size_h &&
	    layer->kernel_size_w() == kernel_size_w &&
	    layer->pad_h()         == pad_h         &&
	    layer->pad_w()         == pad_w         &&
	    layer->stride_h()      == stride_h      &&
	    layer->stride_w()      == stride_w      &&
	    layer->dilation_h()    == dilation_h    &&
	    layer->dilation_w()    == dilation_w    &&
	    layer->group()         == group );
  }

  template <typename Dtype>
  void ConvLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
						     int& num_output, 
						     int& kernel_size_h, int& kernel_size_w,
						     int& pad_h, int& pad_w,
						     int& stride_h, int& stride_w,
						     int& dilation_h, int& dilation_w,
						     bool& bias_term,
						     int& group,
						     std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::ConvolutionParameter& conv_param = layer_param.convolution_param();

    num_output = conv_param.num_output();
    bias_term = conv_param.bias_term();

    if (conv_param.has_kernel_h() || conv_param.has_kernel_w()){
      if (conv_param.kernel_size_size() > 0){
	BOOST_LOG_TRIVIAL(error)<< "ConvLayerBuilder: Either kernel_size or kernel_h/w should be specified; not both.";
      }
      kernel_size_h = conv_param.kernel_h();
      kernel_size_w = conv_param.kernel_w();
    }else{
      kernel_size_h = conv_param.kernel_size(0);
      kernel_size_w = conv_param.kernel_size((conv_param.kernel_size_size() == 1)? 0 : 1);
    }
    
    if (conv_param.has_pad_h() || conv_param.has_pad_w()){
      if (conv_param.pad_size() > 0){
	BOOST_LOG_TRIVIAL(error)<< "ConvLayerBuilder: Either pad or pad_h/w should be specified; not both.";
      }
      pad_h = conv_param.pad_h();
      pad_w = conv_param.pad_w();
    }else if (conv_param.pad_size() > 0){
      pad_h = conv_param.pad(0);
      pad_w = conv_param.pad((conv_param.pad_size() == 1)? 0 : 1);
    }else{
      pad_h = 0;
      pad_w = 0;
    }

    if (conv_param.has_stride_h() || conv_param.has_stride_w()){
      if (conv_param.stride_size() > 0){
	BOOST_LOG_TRIVIAL(error)<< "ConvLayerBuilder: Either stride or stride_h/w should be specified; not both.";
      }
      stride_h = conv_param.stride_h();
      stride_w = conv_param.stride_w();
    }else if(conv_param.stride_size() > 0){
      stride_h = conv_param.stride(0);
      stride_w = conv_param.stride((conv_param.stride_size() == 1)? 0 : 1);
    }else{
      stride_h = 1;
      stride_w = 1;
    }

    // no has_dilation_h and has_dialation_w
    if (conv_param.dilation_size() > 0){
      dilation_h = conv_param.dilation(0);
      dilation_w = conv_param.dilation((conv_param.dilation_size() == 1)? 0 : 1);
    }else{
      dilation_h = 1;
      dilation_w = 1;
    }

    if (conv_param.has_group()){
      group = conv_param.group();
    }else{
      group = 1;
    }
  }
   
} //namespace facethink
