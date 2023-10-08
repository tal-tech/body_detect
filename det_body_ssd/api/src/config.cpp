#include "config.hpp"

namespace facethink {
  namespace detbodyssd {

	  void Config::ReadIniFile(const std::string& config_file) {

		  boost::property_tree::ptree pt;
		  boost::property_tree::ini_parser::read_ini(config_file, pt);

		  //det_config
		  BODY_BOX_THRESHOLD = pt.get<float>("body_box_threshold", BODY_BOX_THRESHOLD);

		  boost::log::add_file_log
		  (
			  boost::log::keywords::auto_flush = true,
			  boost::log::keywords::file_name = "det_body_ssd.log",
			  boost::log::keywords::format = "[%TimeStamp%]: %Message%"
		  );
		  boost::log::add_common_attributes();

		  int log_level = pt.get<int>("log_level", 4);
		  boost::log::core::get()->set_filter(boost::log::trivial::severity >= log_level);
	  }

}
}
