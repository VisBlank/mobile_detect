#include <stdlib.h>
#include <stdio.h>
#include "string.h"
#include <iostream>
#include <fstream>
#include <vector>

#include <android/log.h>
#include <android/bitmap.h>

#include "in_helper_image.h"
#include "common/common.h"
#include "API_mutilabel/API_mutilabel_pvanet.h"
#include "com_example_jni_geekeye_mobile_detection_1_0_0.h"

#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
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
	int i, j, k, tmp, tTime = 0;
	geekeyelab::RunTimer<double> run;

	char szImgPath[256];
	char savePath[256];
	string text;
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,0.35f,0.7f,0,1,CV_AA);
	const static Scalar colors[] =  {
			CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),
			CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
			CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;


	//bitmap_to_mat
	LOGD("[Detection]bitmap_to_mat...");
	cv::Mat mbgr = bitmap_to_mat(env, obj1);
	IplImage image(mbgr);

	//Predict
	vector< MutiLabelInfo > Res;
	run.start();
	LOGD("[Detection]Predict...");
	int nRet = api_muti_label->Predict( (&image), "android", 0, 0, 0.8, Res );
	run.end();
	tTime = int(run.time()*1000.0+0.5);	//(ms)
	
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

	//save img data
	LOGD("[Detection]save...");
	for(i=0;i<Res.size();i++)  
	{						
		Scalar color = colors[i%8];
		cvRectangle( (&image), cvPoint(Res[i].rect[0], Res[i].rect[1]),
                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

		sprintf(szImgPath, "%.2f %s", Res[i].score, Res[i].label.c_str() );
		text = szImgPath;
		cvPutText( (&image), text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

		LOGD("[save img data]:%d:%s",i,text.c_str());
	}
	sprintf( savePath, "%s/res.jpg", g_model_path.c_str() );
	cvSaveImage( savePath, (&image) );
	LOGD("[cvSaveImage]savePath:%s!!",savePath);

	//send data to jni
	env->SetIntArrayRegion(result, 0, size, outData);
	if (outData) {delete [] outData;outData = 0;}

	//
	return result;
}

#ifdef __cplusplus
}
#endif

