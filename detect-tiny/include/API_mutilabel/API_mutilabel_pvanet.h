/*
 * =====================================================================================
 *
 *       filename:  API_mutilabel.h
 *
 *    description:  mutilabel detect interface
 *
 *        version:  1.0
 *        created:  2016-06-20
 *       revision:  none
 *       compiler:  g++
 *
 *         author:  xiaogao
 *        company:  in66.com
 *
 *      copyright:  2016 itugo Inc. All Rights Reserved.
 *      
 * =====================================================================================
 */


#ifndef _API_MUTILABEL_H_
#define _API_MUTILABEL_H_

#include <vector>

#include "API_mutilabel/API_caffe_pvanet.h"

using namespace std;
using namespace caffe;

struct MutiLabelInfo
{
	string 			label;
    float 			score;
    int 			rect[4];
	vector<float>	feat;
};

class API_MUTI_LABEL
{

/***********************************Common***********************************/
#define IMAGE_SIZE 320	//320:30ms,512:50ms,720:80ms

/***********************************public***********************************/
public:

	/// construct function 
    API_MUTI_LABEL();
    
	/// distruct function
	~API_MUTI_LABEL(void);

	/***********************************Init*************************************/
	int Init( 
		const char* 	KeyFilePath,						//[In]:KeyFilePath
		const int 		binGPU, 							//[In]:USE GPU(1) or not(0)
		const int 		deviceID );							//[In]:GPU ID

	/***********************************Predict**********************************/
	int Predict(
		const unsigned char			*image, 				//[In]:image
		const int 					width, 					//[In]:width
		const int 					height, 				//[In]:height
		const int 					nChannel,				//[In]:nChannel
		float						MutiLabel_T,			//[In]:Prdict_0.8,ReSample_0.9
		vector< MutiLabelInfo >		&Res);					//[Out]:Res

	/***********************************Release**********************************/
	void Release();

/***********************************private***********************************/
private:

	/***********************************Init**********************************/
	int i,j,nRet;
	vector< string > 	dic;
	
	/***********************************Init**********************************/
	API_CAFFE_FasterRCNN_MULTILABEL	api_caffe_FasterRCNN_multilabel;

	/***********************************loadWordDict***********************************/
	void loadWordDict(const char *filePath, vector< string > &labelWords);

	/***********************************ResizeImg***********************************/
	unsigned char* ResizeImg( 
		const unsigned char			*image, 				//[In]:image
		const int 					width, 					//[In]:width
		const int 					height, 				//[In]:height
		const int 					nChannel,				//[In]:nChannel
		float 						&w_ratio, 
		float 						&h_ratio, 
		int							&rWidth, 
		int							&rHeight, 
		int 						MaxLen = IMAGE_SIZE );

	unsigned char* cvImageResize(
		const unsigned char 		*src, 
		int 						width, 
		int 						height, 
		int 						nChannel, 
		int 						scale_w, 
		int 						scale_h);

	/***********************************MergeVOC20classLabel**********************************/
	int MutiLabel_Merge(
		const int 								width, 			//[In]:width
		const int 								height, 		//[In]:height
		float									MutiLabel_T,	//[In]:Prdict_0.8,ReSample_0.9
		vector< FasterRCNNInfo_MULTILABEL >		inImgLabel, 	//[In]:ImgDetail from inImgLabel
		vector< MutiLabelInfo >					&LabelInfo);	//[Out]:LabelInfo

};

#endif

	

