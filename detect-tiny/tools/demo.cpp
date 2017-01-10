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

int squeezenet_check( char* modelFile, char* imageFile )
{
	//
	std::string model_file(modelFile);

	//
	GeekeyeDLL ge;
	ge.init(model_file, 0, 0);
	std::vector<float> mean_value(3, 0.0f);
	mean_value[0] = 104;
    mean_value[1] = 117;
    mean_value[2] = 123;
    ge.set_mean_value(mean_value);
    
	//
	std::string image_file(imageFile);
	std::vector< std::pair<int, float> > results;
	ge.predict(image_file, results);
	std::cout<<results[0].first<<" "<<results[0].second<<std::endl;
	
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
		if (img) {cvReleaseImage(&img);img = 0;}
		return -1;
	}

	//
	unsigned char *img_data = Image_Change( img, width, height, nChannel );

	//
	//cout<<"start predict..."<<endl;
	vector< MutiLabelInfo > Res;
	nRet = api_muti_label.Predict( img_data, width, height, nChannel, 0.8, Res );
	if ( (nRet!=0) || (Res.size()<1) )
	{
		printf("[Predict Err]:nRet:%d,results.size():%ld!!\n",nRet,Res.size());
		if (img) {cvReleaseImage(&img);img = 0;}
		if (img_data) {delete [] img_data;img_data = NULL;}
		return -1;
	}
	std::cout<<Res[0].label<<" "<<Res[0].score<<" "
		<<Res[0].rect[0]<<" "<<Res[0].rect[1]<<" "<<Res[0].rect[2]<<" "<<Res[0].rect[3]<<std::endl;

	api_muti_label.Release();
	if (img) {cvReleaseImage(&img);img = 0;}
	if (img_data) {delete [] img_data;img_data = NULL;}
	cout<<"Done!! "<<endl;
	
	return 0;	
}


int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];

	if (argc == 4 && strcmp(argv[1],"-squeezenet_check") == 0) {
		ret = squeezenet_check( argv[2], argv[3] );
	}
	else if (argc == 4 && strcmp(argv[1],"-detect_check") == 0) {
		ret = detect_check( argv[2], argv[3] );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo -squeezenet_check modelFile imageFile\n" << endl;
		cout << "\tDemo -detect_check KeyFilePath imageFile\n" << endl;
		return ret;
	}
	return ret;
}

