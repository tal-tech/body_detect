//#include <vld.h>
#include "det_body_ssd.hpp"
#ifdef WIN32
#pragma comment( lib,"winmm.lib" )
#else
typedef unsigned long       DWORD;
#include <time.h> 
#include <dlfcn.h> 
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#endif

#include "PluginInterface.h"
#include "PluginMgrInterface.h"
#include "base64.h"
#include "TinyThreadPool.h"



#include "json/document.h"  
#include "json/writer.h"  

#include "File.h"

using namespace rapidjson;
using namespace facethink;
using namespace base::Thread;
using namespace base::base64;
using namespace base::CLock;
using namespace base::File;
using namespace std;
using namespace chrono;

typedef int(*pInitfuction)(Plugin::IPluginMgr*, Plugin::IPlugin**, char*, char*); //定义一个函数指针类型
typedef int(*pUninitfuction)(); //定义一个函数指针类型


TinyThreadPool g_pool;

class threadParam
{
public:
	threadParam() {};
	~threadParam() {};
	
	std::string strJson;
	std::string strData;
	Plugin::IPlugin* pPlugin;
	std::string strPtr;
};

unsigned long threadFun(void* param)
{
	threadParam* p = (threadParam*)param;
	if (p)
	{
		char *pResult = new char[1024 * 8];
		std::map<std::string, std::string> mapIn;
		std::map<std::string, std::string> mapOut;
		p->pPlugin->InvokeMethod("body.detect.single",p->strJson.c_str(), pResult);
		
		std::cout << "result: " << pResult << std::endl;
		delete pResult;
	}
	delete p;
	return 0;
}




int testPlugin(int argc, char *argv[])
{

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0]
			<< " det_model"
			<< " image path"
			<< " config file path" << std::endl;
		return 1;
	}

	const std::string det_model = argv[1];
	const std::string image_path = argv[2];
	const std::string config_file_path = argv[3];

	
#ifdef WIN32
	void* hModule = LoadLibrary("det_body_ssd.dll");
	/*
	std::string strCurrentpath = FileUtil::getCurrentAppPath();
	std::string str_det_model = strCurrentpath + std::string("det_body_ssd") + std::string("\\data\\models\\det_body_ssd_v1.0.0.bin");
	std::string  str_config_file = strCurrentpath + std::string("det_body_ssd") + std::string("\\data\\config.ini");
	*/
#else
	void* hModule = dlopen("./libdet_body_ssd.so", RTLD_LAZY);
	/*
	std::string strCurrentpath = FileUtil::getCurrentAppPath();
	std::string str_det_model = strCurrentpath + std::string("det_body_ssd") + std::string("/data/models/det_body_ssd_v1.0.0.bin");
	std::string  str_config_file = strCurrentpath + std::string("det_body_ssd") + std::string("/data/config.ini");
	*/
#endif

  

	if (!hModule)

	{

		std::cout << "Error!" << std::endl;
		return -1;

	}
#ifdef WIN32
	pInitfuction Init = (pInitfuction)GetProcAddress(hModule, "Initialize");
	pUninitfuction Uninit = (pUninitfuction)GetProcAddress(hModule, "UnInitialize");
#else
	pInitfuction Init = (pInitfuction)dlsym(hModule, "Initialize");

	pUninitfuction Uninit = (pUninitfuction)dlsym(hModule, "UnInitialize");

#endif

	if(!Init||!Uninit)
	{
		std::cout << "Get Fuction Error!" << std::endl;
		return -1;
	}

	//MirrorPluginMgr* pMirrMgr = new MirrorPluginMgr();

	std::cout << det_model << std::endl;
	std::cout << config_file_path << std::endl;

	Plugin::IPlugin* pPlugin = NULL;
	Init(NULL, &pPlugin, (char*)det_model.c_str(), (char*)config_file_path.c_str());
	std::cout << "init succeed" << std::endl;


	std::map<std::string, std::string> inParam;
	std::map<std::string, std::string> outParam;


	std::vector<uchar> buf = FileUtil::readFromFile(image_path.c_str());



	for (int i = 0; i < 1000; i++)
	{

		int base64_len = calc_base64_len(buf.size());
		char * pOut = new char[base64_len+1];
		memset(pOut, 0, sizeof(char)*(base64_len + 1));

		std::string str(buf.begin(), buf.end());
		const char* pData = str.c_str();

		auto start1 = system_clock::now();
		base64_encode(pData, buf.size(), pOut);
		auto end1 = system_clock::now();
		std::cout << "base 64 encode cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms" << std::endl;
		inParam["img"] = std::string(pOut);

		cv::Mat img = cv::imread(image_path); //test image
		if (img.data == 0) {
			return -1;
		}


		Document doc;
		doc.SetObject();    //key-value 相当与map
							//doc.Setvalue();        //数组型 相当与vector
		Document::AllocatorType &allocator = doc.GetAllocator(); //获取分配器

																 //2，给doc对象赋值
		doc.AddMember("img", rapidjson::Value(pOut, allocator), allocator);
		StringBuffer buffer;
		Writer<StringBuffer> writer(buffer);
		doc.Accept(writer);
		for (int i = 0; i < 1; i++)
		{
			std::string out;
			char *pResult = new char[1024 * 8];

			auto start = system_clock::now();
			pPlugin->InvokeMethod("body.detect.single", buffer.GetString(), pResult);
			auto end = system_clock::now();
			std::cout << "image analyze cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

			threadParam *pParam = new threadParam();
			pParam->strData = pOut;// std::string(buffer.GetString());
			pParam->strJson = std::string(buffer.GetString());
			pParam->pPlugin = pPlugin;


			ThreadToolParam *pThreadParam = new ThreadToolParam();
			pThreadParam->pWork = threadFun;
			pThreadParam->pThreadFuncParam = pParam;
			g_pool.Add(pThreadParam);
		}
		delete[] pOut;
	}


	return -1;
}




int main(int argc, char *argv[]) {
	g_pool.Open(10, 20);
	testPlugin(argc, argv);
	getchar();
	return 0;
}
