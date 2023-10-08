#pragma once
#include "PluginInterface.h"
#include <map>
#include <string>
#include "det_body_ssd.hpp"
#include "Singleton.h"
#include "CLock.h"

class FaceBodyPlugin :public Plugin::IPlugin
{
public:
	FaceBodyPlugin();
	~FaceBodyPlugin();
public:
	facethink::BodyDetectionSSD * m_body_detector_server;
	char * m_pBuffer1;
	char * m_pBuffer2;
	char * m_pBuffer3;
	base::CLock::CCriticalSection m_cs;
public:

	int InvokeMethod(const char* strCommand,
		const  std::map<std::string, std::string>& args,
		std::map<std::string, std::string>& out
	);


	int InvokeMethod(const char* strCommand,
		const char* args,
		char* out
	);


	int GetInterface(
		const std::map<std::string, std::string>& args,
		std::map<std::string, std::string>& out
	);

};

