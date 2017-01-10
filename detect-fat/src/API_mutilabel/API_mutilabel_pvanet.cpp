//#pragma once
#include <queue>  // for std::priority_queue
#include <utility>  // for pair

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <dirent.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <time.h>
#include <sys/mman.h> /* for mmap and munmap */
#include <sys/types.h> /* for open */
#include <sys/stat.h> /* for open */
#include <fcntl.h>     /* for open */
#include <pthread.h>

#include <vector>
#include <list>
#include <map>
#include <algorithm>

#include "API_mutilabel/API_mutilabel_pvanet.h"

using namespace std;

#define CHECK 1	//check release model

static bool Sort_Info(const MutiLabelInfo& elem1, const MutiLabelInfo& elem2)
{
    return (elem1.score > elem2.score);
}

/***********************************Init*************************************/
/// construct function 
API_MUTI_LABEL::API_MUTI_LABEL()
{
}

/// destruct function 
API_MUTI_LABEL::~API_MUTI_LABEL(void)
{
}

/***********************************Init*************************************/
int API_MUTI_LABEL::Init( 
	const char* 	KeyFilePath,						//[In]:KeyFilePath
	const int		binGPU, 							//[In]:USE GPU(1) or not(0)
	const int		deviceID )	 						//[In]:GPU ID
{
	char tPath[1024] = {0};
	char tPath2[1024] = {0};
	char tPath3[1024] = {0};

	string strLayerName;
	nRet = 0;
	
	/***********************************Init**********************************/
	//PVANet-lite
	strLayerName = "cls_prob";

	//double
	sprintf(tPath, "%s/mutilabel/v2.1.0/test_proposal20.pt",KeyFilePath);
	sprintf(tPath2, "%s/mutilabel/v2.1.0/pvanet_frcnn_iter_100w.caffemodel",KeyFilePath); //55.35%
#ifdef CHECK	//check release model
	nRet = api_caffe_FasterRCNN_multilabel.init_release( tPath, tPath2, strLayerName.c_str(), binGPU, deviceID ); 
#else
	nRet = api_caffe_FasterRCNN_multilabel.Init( tPath, tPath2, strLayerName.c_str(), binGPU, deviceID ); 
#endif

	//uint8
	//int type = 1;	//type:1-Uint8,0-Dtype;
	//sprintf(tPath, "%s/mutilabel/v2.1.0/pvanet_frcnn_iter_100w_convert.caffemodel",KeyFilePath); //55.35%
	//nRet = api_caffe_FasterRCNN_multilabel.Init( tPath, strLayerName.c_str(), type, binGPU, deviceID ); 
	
	if (nRet != 0)
	{
	   //LOOGE<<"Fail to initialization ";
	   cout<<"Fail to initialization "<<endl;
	   return -1;
	}
	
	/***********************************Load dic File**********************************/
	dic.clear();
	sprintf(tPath, "%s/mutilabel/v2.1.0/Dict_mutilabel.txt",KeyFilePath);
	printf("load dic:%s\n",tPath);
	loadWordDict(tPath,dic);
	printf( "dict:size-%d,tag:", int(dic.size()) );
	for ( i=0;i<dic.size();i++ )
	{
		printf( "%d-%s ",i,dic[i].c_str() );
	}
	printf( "\n" );

	return nRet;
}

/***********************************loadWordDict***********************************/
void API_MUTI_LABEL::loadWordDict(const char *filePath, vector< string > &labelWords)
{
	ifstream ifs(filePath);
	if(!ifs)
	{
		printf("open %s failed.\n", filePath);
		assert(false);
	}
	string line ;
	while(getline(ifs, line))
	{
		if(line.length() != 0)
			labelWords.push_back(line);
	}
	assert(labelWords.size());
}

int API_MUTI_LABEL::Predict(
	const unsigned char			*image, 				//[In]:image
	const int 					width, 					//[In]:width
	const int 					height, 				//[In]:height
	const int 					nChannel,				//[In]:nChannel
	float						MutiLabel_T,		//[In]:Prdict_0.8,ReSample_0.9
	vector< MutiLabelInfo >		&Res)				//[In]:Layer Name by Extract
{	
	if(!image || (width<16) || (height<16) ) 
	{	
		//LOOGE<<"input err!!";
		cout<<"input err!!"<<endl;
		return 101;
	}

	int i,x1,y1,x2,y2,topN = 100;
	nRet = 0;
	vector<FasterRCNNInfo_MULTILABEL>		vecLabel;
	vector<MutiLabelInfo> 		MergeRes;
	Res.clear();

	//Send Err Info
	{
		MutiLabelInfo errInfo;
		errInfo.label = "other.other";
		errInfo.score = 1.0;
		errInfo.rect[0] = 0;
		errInfo.rect[1] = 0;
		errInfo.rect[2] = width-1;
		errInfo.rect[3] = height-1;
		errInfo.feat.clear();
		Res.push_back( errInfo );
	}

	/************************ResizeImg*****************************/
	float w_ratio = 1.0;
	float h_ratio = 1.0;
	int rWidth = 0;
	int rHeight = 0;
	//320:30ms,512:50ms,720:80ms
	unsigned char* imgResize = ResizeImg( image, width, height, nChannel, w_ratio, h_ratio, rWidth, rHeight, IMAGE_SIZE );
	//if(!imgResize || imgResize->nChannels != 3 || imgResize->depth != IPL_DEPTH_8U) 
	if(!imgResize) 
	{	
		//LOOGE<<"Fail to ResizeImg";
		cout<<"Fail to ResizeImg"<<endl;
		if (imgResize) {delete [] imgResize;imgResize = NULL;}
		return 102;
	}
	
	/***********************************Predict**********************************/
	cout<<"start to api_caffe_FasterRCNN_multilabel.Predict..."<<endl;
	vecLabel.clear();
#ifdef CHECK	//check release model
	nRet = api_caffe_FasterRCNN_multilabel.Predict_release( imgResize, rWidth, rHeight, nChannel, MutiLabel_T, vecLabel );
#else
	nRet = api_caffe_FasterRCNN_multilabel.Predict( imgResize, rWidth, rHeight, nChannel, MutiLabel_T, vecLabel );
#endif
	if ( (nRet!=0) || (vecLabel.size()<1) )
	{
	   //LOOGE<<"Fail to Predict";
	   cout<<"Fail to Predict"<<endl;
	   if (imgResize) {delete [] imgResize;imgResize = NULL;}
	   return 103;
	}

	/************************MutiLabel_Merge*****************************/
	cout<<"start to MutiLabel_Merge..."<<endl;
	MergeRes.clear();
	nRet = MutiLabel_Merge( rWidth, rHeight, MutiLabel_T, vecLabel, MergeRes );
	if ( (nRet!=0) || (MergeRes.size()<1) )
	{
		//LOOGE<<"[MutiLabel_Merge Err!!]";
		cout<<"[MutiLabel_Merge Err!!]"<<endl;
		if (imgResize) {delete [] imgResize;imgResize = NULL;}
		return 104;
	}

	/************************MutiLabel_Merge Res*****************************/
	Res.clear();
	topN = (MergeRes.size()>topN)?topN:MergeRes.size();
	if ( ( w_ratio == 1.0 ) && ( h_ratio == 1.0 ) )
	{
		Res.assign( MergeRes.begin(), MergeRes.begin()+topN );
	}
	else
	{
		for(i=0;i<topN;i++)
		{
			x1 = int(MergeRes[i].rect[0]*1.0/w_ratio + 0.5);
			y1 = int(MergeRes[i].rect[1]*1.0/h_ratio + 0.5);
			x2 = int(MergeRes[i].rect[2]*1.0/w_ratio + 0.5);
			y2 = int(MergeRes[i].rect[3]*1.0/h_ratio + 0.5);
			MutiLabelInfo ratioInfo;
			ratioInfo.label = MergeRes[i].label;
			ratioInfo.score = MergeRes[i].score;
			ratioInfo.rect[0] = x1;
			ratioInfo.rect[1] = y1;
			ratioInfo.rect[2] = x2;
			ratioInfo.rect[3] = y2;
			std::copy(MergeRes[i].feat.begin(),MergeRes[i].feat.end(), std::back_inserter(ratioInfo.feat)); 
			Res.push_back( ratioInfo );
		}
	}

	if (imgResize) {delete [] imgResize;imgResize = NULL;}

	return nRet;
}

unsigned char* API_MUTI_LABEL::ResizeImg( 
		const unsigned char 		*image, 				//[In]:image
		const int					width,					//[In]:width
		const int					height, 				//[In]:height
		const int					nChannel,				//[In]:nChannel
		float						&w_ratio, 
		float						&h_ratio, 
		int							&rWidth, 
		int							&rHeight, 
		int 						MaxLen )
{
	int tmp, nRet = 0;
	int stride = 32;
	w_ratio = 1.0;
	h_ratio = 1.0;

	rWidth = 0;
	rHeight = 0;

	if ( MaxLen%stride != 0 )
	{
		printf("[API_COMMEN::ResizeImg]MaxLen:%d,stride:%d\n",MaxLen,stride);
		return NULL;
	}

	//Resize
	if (width > height) {
		w_ratio = MaxLen*1.0 / width;
		tmp = (int )height * w_ratio;
		tmp = (((tmp + 16) >> 5) << 5);
		h_ratio = tmp*1.0 / height;

		rWidth = MaxLen;
		rHeight = tmp;
	} 
	else 
	{	
		h_ratio = MaxLen*1.0 / height;
		tmp = (int )width * h_ratio;
		tmp = (((tmp + 16) >> 5) << 5);
		w_ratio = tmp*1.0 / width;

		rWidth = tmp;
		rHeight = MaxLen;
	}

	//printf("im_info:%d,%d,%d,%d,%.2f,%.2f\n",img->width,img->height,rWidth,rHeight,w_ratio,h_ratio);

	unsigned char* imgResize = cvImageResize( image, width, height, nChannel, rWidth, rHeight );

	return imgResize;
}

unsigned char* API_MUTI_LABEL::cvImageResize(
	const unsigned char 		*src, 
	int 						width, 
	int 						height, 
	int 						nChannel, 
	int 						scale_w, 
	int 						scale_h)
{
	if( (!src) || (width<16) || (height<16) || (scale_w<1) || (scale_h<1) )
		return NULL;

	int i,j,k,scale_x,scale_y,index=0;
	int nSize = scale_w * scale_h * nChannel;
	unsigned char *dst = new unsigned char[nSize];

	for (i=0; i<scale_h; ++i)
    {
    	scale_y = int(i*height*1.0/scale_h+0.5);
        for (j=0; j<scale_w; ++j)
        {
        	scale_x = int(j*width*1.0/scale_w+0.5);
        	for (k=0; k<nChannel; ++k)
        	{
        		index = scale_y*width*nChannel+scale_x*nChannel+k;
				dst[i*scale_w*nChannel+j*nChannel+k] = src[index];
        	}
		}
	}

	return dst;
}


int API_MUTI_LABEL::MutiLabel_Merge(
		const int 								width, 			//[In]:width
		const int 								height, 		//[In]:height
		float									MutiLabel_T,	//[In]:Prdict_0.8,ReSample_0.9
		vector< FasterRCNNInfo_MULTILABEL >		inImgLabel, 	//[In]:ImgDetail from inImgLabel
		vector< MutiLabelInfo >					&LabelInfo)		//[Out]:LabelInfo
{
	if ( inImgLabel.size() < 1 ) 
	{ 
		//LOOGE<<"MergeLabel[err]:inImgLabel.size()<1!!";
		cout<<"MergeLabel[err]:inImgLabel.size()<1!!"<<endl;
		return -1;
	}
	
	int i,k,label,BinMode,bin_sv_Filter_Face;
	float score = 0.0;
	nRet=0;
	LabelInfo.clear();

	for ( i=0;i<inImgLabel.size();i++ )
	{
		label = inImgLabel[i].label;
		score = inImgLabel[i].score;		

		if ( (label<dic.size()) && (score>=MutiLabel_T) )
		{
			MutiLabelInfo mutiLabelInfo;
			mutiLabelInfo.label = dic[label];
			mutiLabelInfo.score = score;
			mutiLabelInfo.rect[0] = inImgLabel[i].rect[0];
			mutiLabelInfo.rect[1] = inImgLabel[i].rect[1];
			mutiLabelInfo.rect[2] = inImgLabel[i].rect[2];
			mutiLabelInfo.rect[3] = inImgLabel[i].rect[3];
			std::copy(inImgLabel[i].feat.begin(),inImgLabel[i].feat.end(), std::back_inserter(mutiLabelInfo.feat)); 
			
			LabelInfo.push_back( mutiLabelInfo );
		}
	}

	//Send Err Info
	if (LabelInfo.size()<1)
	{
		MutiLabelInfo errInfo;
		errInfo.label = "other.other";
		errInfo.score = 0;
		errInfo.rect[0] = 0;
		errInfo.rect[1] = 0;
		errInfo.rect[2] = width-1;
		errInfo.rect[3] = height-1;
		errInfo.feat.clear();
		LabelInfo.push_back( errInfo );
	}
	else
	{
		std::sort(LabelInfo.begin(),LabelInfo.end(),Sort_Info);
	}
	
	return 0;
}

/***********************************Release**********************************/
void API_MUTI_LABEL::Release()
{
	/***********************************net Model**********************************/
#ifdef CHECK	//check release model
	api_caffe_FasterRCNN_multilabel.Release_release();
#else
	api_caffe_FasterRCNN_multilabel.Release();
#endif
}


