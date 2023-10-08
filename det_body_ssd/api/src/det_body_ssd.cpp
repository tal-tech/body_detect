#include "frame_state_detector.hpp"
#include "det_body_ssd.hpp"
#include "FaceBodyPlugin.h"
#include "PluginMgrInterface.h"
#include <string>
#include "Singleton.h"
using namespace base::CSingleton;

#ifdef WIN32  
#include <Windows.h>  
#else  
#include <stdio.h>  
#include <unistd.h>  
#endif  

namespace facethink {
  using namespace detbodyssd;
	
	BodyDetectionSSD::BodyDetectionSSD(void) {

	}

	BodyDetectionSSD::~BodyDetectionSSD(void) {

	}

	BodyDetectionSSD* BodyDetectionSSD::create(
		const std::string& det_model_prototxt,
		const std::string& det_model_binary,
		const std::string& config_file,
		const int gpu_id){

		return new FrameStateDetector(
			det_model_prototxt,
			det_model_binary,
		    config_file,
			gpu_id);
	}

	BodyDetectionSSD* BodyDetectionSSD::create(
		const std::string& det_model_file,
		const std::string& config_file,
		const int gpu_id) {

		return new FrameStateDetector(
			det_model_file,
			config_file,
			gpu_id);
	}

}

Plugin::IPluginMgr* gPluginMgr = NULL;

std::string getCurrentAppPath()
{
#ifdef WIN32  
	char path[MAX_PATH + 1] = { 0 };
	if (GetModuleFileName(NULL, path, MAX_PATH) != 0)
	{
		std::string strPath(path);
		std::size_t found = strPath.find_last_of("\\");
		std::string retStr = strPath.substr(0, found + 1);
		return retStr;
	}
#else  
	char path[256] = { 0 };
	char filepath[256] = { 0 };
	char cmd[256] = { 0 };
	FILE* fp = NULL;

	// 设置进程所在proc路径  
	sprintf(filepath, "/proc/%d", getpid());
	// 将当前路径设为进程路径  
	if (chdir(filepath) != -1)
	{
		//指定待执行的shell 命令  
		snprintf(cmd, 256, "ls -l | grep exe | awk '{print $10}'");
		if ((fp = popen(cmd, "r")) == NULL)
		{
			return std::string();
		}
		//读取shell命令执行结果到字符串path中  
		if (fgets(path, sizeof(path) / sizeof(path[0]), fp) == NULL)
		{
			pclose(fp);
			return std::string();
		}

		//popen开启的fd必须要pclose关闭  
		pclose(fp);


		std::string strPath(path);
		std::size_t found = strPath.find_last_of("/");
		std::string retStr = strPath.substr(0, found + 1);
		return retStr;

		//return std::string(path);
	}
#endif  


	return std::string();
}



void InitModel(char*  det_model, char*  config_file)
{
	std::string str_det_model;
	std::string str_config_file;
	if(!det_model|| !config_file)
	{
		std::string strCurrentpath = getCurrentAppPath();
		str_det_model = strCurrentpath+std::string("data\\models\\det_body_ssd_v1.0.0.bin");
		str_config_file = strCurrentpath + std::string("data\\config.ini");
	}

	str_det_model = det_model;
	str_config_file = config_file;

	facethink::BodyDetectionSSD *body_detector_server = facethink::BodyDetectionSSD::create(
		str_det_model,
		str_config_file);
	FaceBodyPlugin* pPlugin = CSingleton<FaceBodyPlugin>::getInstance();
	pPlugin->m_body_detector_server = body_detector_server;
	
}

/*插件初始化时调用的函数，用以交换插件平台与插件的指针*/

EXPORT_API int  Initialize(

	Plugin::IPluginMgr* pPluginMgr,
	Plugin::IPlugin** ppPlugin,
	char*  det_model,
	char*  config_file)
{
 	if (pPluginMgr = NULL)
 		return 1;	
	InitModel(det_model, config_file);
	gPluginMgr = pPluginMgr;

 	*ppPlugin = CSingleton<FaceBodyPlugin>::getInstance();


	return 0;
}



EXPORT_API int  UnInitialize()
{
	gPluginMgr = NULL;
	return 0;
}
