#ifndef __FACETHINK_API_FACE_DETECTION_CONFIG_HPP__
#define __FACETHINK_API_FACE_DETECTION_CONFIG_HPP__

#include <vector>
#include <string>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

namespace facethink {
  namespace detbodyssd {
		class Config {
		public:
			Config() {

				BODY_BOX_THRESHOLD = 0.5;

				boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::error);
			}

		public:
			float BODY_BOX_THRESHOLD;

			void ReadIniFile(const std::string& config_file); // class Config
		};

	}	
}

#endif
