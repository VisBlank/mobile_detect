#include <stdlib.h>
#include <stdio.h>
#include "string.h"
#include <iostream>
#include <fstream>
#include <vector>

#include <android/log.h>
#include <android/bitmap.h>

//#include "in_helper_image.h"
#include "in_bitmap_processor.h"
#include "common/common.h"
#include "API_mutilabel/API_mutilabel_pvanet.h"
#include "com_example_jni_geekeye_mobile_detection_1_0_0.h"

using namespace std;

static API_MUTI_LABEL *api_muti_label;
static string g_model_path;


#define TAG    "log-jni" // 这个是自定义的LOG的标识
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__) // 定义LOGD类型

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_example_jni_geekeye_mobile_detection_1_0_0
 * Method:    DL_Init_Detection
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_com_example_jni_geekeye_1mobile_1detection_11_10_10_DL_1Init_1Detection
  (JNIEnv * env, jobject obj, jstring ModelPath)
{
	int ret = 0;  	
	const char* str_model_path = env->GetStringUTFChars(ModelPath, NULL);	
	g_model_path = string(str_model_path);
	LOGD("model path: %s", str_model_path);

	api_muti_label = new API_MUTI_LABEL();	
	ret = api_muti_label->Init(str_model_path, 0, 0);
	if (ret!=0)
	{
		LOGD("ret:%d,model path:%s", ret, str_model_path);
		return ret;
	}

	LOGD("api_muti_label->Init end!!");

	return ret;
}

/*
 * Class:     com_example_jni_geekeye_mobile_detection_1_0_0
 * Method:    Image_Detection
 * Signature: (Landroid/graphics/Bitmap;)[I
 */
JNIEXPORT jintArray JNICALL Java_com_example_jni_geekeye_1mobile_1detection_11_10_10_DL_1Image_1Detection
  (JNIEnv * env, jobject obj, jobject obj1)
{
	LOGD("[Detection]init1...");
	//init output data
	int size = 3;
	int* outData = new int[size];
	memset(outData, 0, size * sizeof(int));

	jintArray result = env->NewIntArray(size);
	env->SetIntArrayRegion(result, 0, size, outData);

	//init
	LOGD("[Detection]init2...");
	int i, j, k, tmp, width, height, nChannel, nRet, tTime = 0;
	geekeyelab::RunTimer<double> run;

	width = 240;
	height = 320;
	nChannel = 4;

	//get image
	uint8_t* image = NULL;
    nRet = bitmap_resize<uint8_t>(env, obj1, image, width, height);

	//Predict
	vector< MutiLabelInfo > Res;
	run.start();
	LOGD("[Detection]Predict...");
	nRet = api_muti_label->Predict( image, width, height, nChannel, 0.8, Res );
	run.end();
	tTime = int(run.time()*1000.0+0.5);	//(ms)
	free(image);
	
	//write data
	outData[0] = nRet;
	outData[1] = tTime;
	outData[2] = Res.size();
		
	if ( (nRet!=0) || (Res.size()<1) )
	{
		LOGD("[Predict]err:%d,time:%d(ms)!!",nRet,tTime);

		env->SetIntArrayRegion(result, 0, size, outData);
		if (outData) {delete [] outData;outData = 0;}
		return result;
	}

	//log
	char tmpPath[1024];
	sprintf( tmpPath, "res[0]:%s_%.4f_%d_%d_%d_%d", Res[0].label.c_str(),Res[0].score,
		Res[0].rect[0],Res[0].rect[1],Res[0].rect[2],Res[0].rect[3]);
	LOGD("%s",tmpPath);

	//send data to jni
	env->SetIntArrayRegion(result, 0, size, outData);
	if (outData) {delete [] outData;outData = 0;}

	//
	return result;
}

#ifdef __cplusplus
}
#endif

