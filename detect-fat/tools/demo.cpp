#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>		//do shell
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <map>

//#include "cv.h"
//#include "cxcore.h"
//#include "highgui.h"
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "caffe/caffe.hpp"
#include "geekeyedll/geekeye_dll.h"
#include "common/common.h"

#include "API_mutilabel/API_mutilabel_pvanet.h"

using namespace cv;
using namespace std;
using namespace geekeyelab;

#define CHECK 1	//check release model

int model_convert_release( char *file_deploy, char* file_model, char* file_sv )
{
	std::string deploy_file(file_deploy);
	std::string model_file(file_model);
	std::string sv_file(file_sv);

	//
	GeekeyeDLL ge;
	ge.init(deploy_file, model_file);

	//
	ge.save_net(sv_file);
	
  	//
	return 0;
}


/***********************************GetIDFromFilePath***********************************/
static string GetStringIDFromFilePath(const char *filepath)
{
	long ID = 0;
	int  atom =0;
	string tmpPath = filepath;
	string iid;

	long start = tmpPath.find_last_of('/');
	long end = tmpPath.find_last_of('.');

	if ( (start>0) && (end>0) && ( end>start ) )
		iid = tmpPath.substr(start+1,end-start-1);
	
	return iid;
}

unsigned char * Image_Change(IplImage *src, int &width, int &height, int &nChannel )
{
	if(!src || (src->width<16) || (src->height<16))
    {
        cvReleaseImage(&src);src = 0;
        return NULL;
    }

	int i,j,k;
	width = src->width;
	height = src->height;
	nChannel = src->nChannels;
	int nSize = width * height * nChannel;
	unsigned char *dst = new unsigned char[nSize];

	for (i=0; i<height; ++i)
    {
        for (j=0; j<width; ++j)
        {
        	for (k=0; k<nChannel; ++k)
        	{
				dst[i*width*nChannel+j*nChannel+k] = 
					((unsigned char *)(src->imageData + i*src->widthStep))[j*src->nChannels + k];
        	}
		}
	}

	return dst;
}

int squeezenet_check( char *deployFile, char* modelFile, char* imageFile )
{
	//
	std::string deploy_file(deployFile);
	std::string model_file(modelFile);

	//
	GeekeyeDLL ge;
#ifdef CHECK	//check release model
	ge.init_release(deploy_file,model_file);
#else
	ge.init(deploy_file,model_file);
#endif

	std::vector<float> mean_value(3, 0.0f);
	mean_value[0] = 104;
    mean_value[1] = 117;
    mean_value[2] = 123;
    ge.set_mean_value(mean_value);
    
	//
	IplImage *img = cvLoadImage(imageFile);
	if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
	{	
		cout<<"Can't open " << imageFile << endl;
		cvReleaseImage(&img);img = 0;
		return -1;
	}

	//
	IplImage *imgResize = cvCreateImage(cvSize(224, 224), img->depth, img->nChannels);
	cvResize( img, imgResize );

	//
	int width,height,nChannel;
	unsigned char *img_data = Image_Change( imgResize, width, height, nChannel );

	//
	cout<<"start predict..."<<endl;
	std::vector< std::pair<int, float> > results;
	int nRet = ge.predict(img_data, width, height, results);
	if ( (nRet!=0) || (results.size()<1) )
	{
		printf("[Predict Err]:nRet:%d,results.size():%ld!!\n",nRet,results.size());
		cvReleaseImage(&img);img = 0;
		cvReleaseImage(&imgResize);imgResize = 0;
		if (img_data) {delete [] img_data;img_data = NULL;}
		return -1;
	}
	std::cout<<results[0].first<<" "<<results[0].second<<std::endl;

	cvReleaseImage(&img);img = 0;
	cvReleaseImage(&imgResize);imgResize = 0;
	if (img_data) {delete [] img_data;img_data = NULL;}
	cout<<"Done!! "<<endl;
	
	return 0;	
}

int detect_check( char *KeyFilePath, char* imageFile )
{
	int nRet,width,height,nChannel;
	API_MUTI_LABEL api_muti_label;
	nRet = api_muti_label.Init( KeyFilePath, 0, 0 );
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return -1;
	}
    
	//
	IplImage *img = cvLoadImage(imageFile);
	if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
	{	
		cout<<"Can't open " << imageFile << endl;
		cvReleaseImage(&img);img = 0;
		return -1;
	}

	//
	unsigned char *img_data = Image_Change( img, width, height, nChannel );

	//
	cout<<"start predict..."<<endl;
	vector< MutiLabelInfo > Res;
	nRet = api_muti_label.Predict( img_data, width, height, nChannel, 0.8, Res );
	if ( (nRet!=0) || (Res.size()<1) )
	{
		printf("[Predict Err]:nRet:%d,results.size():%ld!!\n",nRet,Res.size());
		cvReleaseImage(&img);img = 0;
		if (img_data) {delete [] img_data;img_data = NULL;}
		return -1;
	}
	std::cout<<Res[0].label<<" "<<Res[0].score<<" "
		<<Res[0].rect[0]<<" "<<Res[0].rect[1]<<" "<<Res[0].rect[2]<<" "<<Res[0].rect[3]<<std::endl;

	api_muti_label.Release();
	cvReleaseImage(&img);img = 0;
	if (img_data) {delete [] img_data;img_data = NULL;}
	cout<<"Done!! "<<endl;
	
	return 0;	
}


int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];

	if (argc == 5 && strcmp(argv[1],"-model_convert_release") == 0) {
		ret = model_convert_release( argv[2], argv[3], argv[4] );
	}
	else if (argc == 5 && strcmp(argv[1],"-squeezenet_check") == 0) {
		ret = squeezenet_check( argv[2], argv[3], argv[4] );
	}
	else if (argc == 4 && strcmp(argv[1],"-detect_check") == 0) {
		ret = detect_check( argv[2], argv[3] );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo -model_convert file_deploy file_model file_sv\n" << endl;
		cout << "\tDemo -squeezenet_check deployFile modelFile imageFile\n" << endl;
		cout << "\tDemo -detect_check KeyFilePath imageFile\n" << endl;
		return ret;
	}
	return ret;
}

