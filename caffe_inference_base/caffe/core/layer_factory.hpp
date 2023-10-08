#ifndef __FACETHINK_CAFFE_CORE_LAYER_FACTORY_HPP__
#define __FACETHINK_CAFFE_CORE_LAYER_FACTORY_HPP__

#include <map>
#include <string>
#include <functional>
#include "caffe/core/base_layer_builder.hpp"

namespace facethink {
  
  template <typename Dtype>
  class LayerFactory {
  public:

    template <typename T>
    struct register_t{
      register_t(const std::string & key) {
	if (LayerFactory<Dtype>::instance().map_.find(key) ==
	    LayerFactory<Dtype>::instance().map_.end()) {
	  BOOST_LOG_TRIVIAL(debug)<< "LayerFactory: Registering LayerBuilder ["<< key << "].";
	  LayerFactory<Dtype>::instance().map_.insert( { key, std::make_shared<T>() } );
	}
      }
    };

    static LayerFactory& instance() {
      static LayerFactory<Dtype> instance;
      return instance;
    }
    
    const std::shared_ptr<BaseLayerBuilder<Dtype> > get_builder(const std::string & key) {
      if(map_.find(key) == map_.end()) {
	BOOST_LOG_TRIVIAL(fatal)<< "LayerFactory: LayerBuilder ["<< key << "] doesn't exist.";
	return nullptr;
      }
      return map_[key];
    }
    
  private:
    LayerFactory() {};
    LayerFactory(const LayerFactory &) = delete;
    LayerFactory(LayerFactory &&) = delete;
  
    std::map<std::string, std::shared_ptr<BaseLayerBuilder<Dtype> > > map_;
  }; // class layerfactory

  
#define REGISTER_LAYER_BUILDER_FULL(Dtype, T, name, key)	\
  static LayerFactory<Dtype>::register_t<T> name(key)

#define REGISTER_LAYER_BUILDER_TMP(Dtype, T, key) \
  REGISTER_LAYER_BUILDER_FULL(Dtype, T, key##_layer_builder_, #key)

#define REGISTER_LAYER_BUILDER(T, key) \
  REGISTER_LAYER_BUILDER_TMP(float, T<float>, key) 

    
} // namespace facethink



#endif 
