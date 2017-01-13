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

#include "API_mutilabel/API_mutilabel_pvanet.h"
#include "common/common.h"

using namespace cv;
using namespace std;

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

int frcnn_test( char *szQueryList, char* svPath, char* KeyFilePath, float MutiLabel_T, int binGPU, int deviceID )
{
	char tPath[256];
	char loadImgPath[256];
	char szImgPath[256];
	char savePath[256];
	vector<string> vecText;
	int i, j, label, svImg, nRet = 0;
	long inputLabel, nCount, nCountObj;
	string strImageID,text,name;
	double allPredictTime;
	FILE *fpListFile = 0;

	API_MUTI_LABEL api_muti_label;

	vector< MutiLabelInfo > Res;

	geekeyelab::RunTimer<double> run;  

	/***********************************Init**********************************/
	//unsigned long long		imageID;				//[In]:image ID for CheckData
	//unsigned long long		childID;				//[In]:image child ID for CheckData
	//sprintf( tPath, "res/plog.log" );
	//plog::init(plog::error, tPath); 
	//sprintf( tPath, "res/module-in-logo-detection.log" );
	//plog::init<enum_module_in_logo>(plog::info, tPath, 100000000, 100000);

	const static Scalar colors[] =  { 	CV_RGB(0,0,255),	CV_RGB(0,255,255),	CV_RGB(0,255,0),	
										CV_RGB(255,255,0),	CV_RGB(255,0,0),	CV_RGB(255,0,255),
										CV_RGB(0,128,255), 	CV_RGB(255,128,0)} ;

	CvFont font;
 	cvInitFont(&font,CV_FONT_HERSHEY_TRIPLEX,0.35f,0.7f,0,1,CV_AA);
	
	/********************************Open Query List*****************************/
	fpListFile = fopen(szQueryList,"r");
	if (!fpListFile) 
	{
		cout << "0.can't open " << szQueryList << endl;
		return -1;
	}

	/***********************************Init*************************************/
	nRet = api_muti_label.Init( KeyFilePath, binGPU, deviceID ); 
	if (nRet != 0)
	{
	   cout<<"Fail to initialization "<<endl;
	   return -1;
	}
	
	printf("Init end!!\n");

	nCount = 0;
	nCountObj = 0;
	allPredictTime = 0.0;
	/*****************************Process one by one*****************************/
	while(EOF != fscanf(fpListFile, "%s", loadImgPath))
	{
		IplImage *img = cvLoadImage(loadImgPath);
		if(!img || (img->width<32) || (img->height<32) || img->nChannels != 3 || img->depth != IPL_DEPTH_8U) 
		{	
			cout<<"Can't open " << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}	
		//printf("loadImgPath:%s\n",loadImgPath);

		/************************getRandomID*****************************/
		strImageID = GetStringIDFromFilePath( loadImgPath );
		
		/************************Predict*****************************/	
		Res.clear();
		run.start();
		sprintf( tPath, "%d_%d", nCount, nCountObj );
		nRet = api_muti_label.Predict( img, string(tPath), nCount, nCountObj, MutiLabel_T, Res );
		if ( (nRet!=0) || (Res.size()<1) )
		{
			//LOOGE<<"[Predict Err!!loadImgPath:]"<<loadImgPath;
			cout<<"[Predict Err!!loadImgPath:]" << loadImgPath << endl;
			cvReleaseImage(&img);img = 0;
			continue;
		}
		run.end();
		//LOOGI<<"[Predict] time:"<<run.time();
		allPredictTime += run.time();
		
		/************************save img data*****************************/
		for(i=0;i<Res.size();i++)  
		{						
			Scalar color = colors[i%8];
			cvRectangle( img, cvPoint(Res[i].rect[0], Res[i].rect[1]),
	                   cvPoint(Res[i].rect[2], Res[i].rect[3]), color, 2, 8, 0);

			sprintf(szImgPath, "%.2f %s", Res[i].score, Res[i].label.c_str() );
			text = szImgPath;
			cvPutText( img, text.c_str(), cvPoint(Res[i].rect[0]+1, Res[i].rect[1]+20), &font, color );

			//if (i<3)
			//	printf("Res:%.2f_%.2f_%.2f!!\n",Res[i].feat[0],Res[i].feat[1],Res[i].feat[2]);
		}
		sprintf( savePath, "%s/%s.jpg", svPath, strImageID.c_str() );
		cvSaveImage( savePath, img );

		nCountObj += Res.size();
		nCount++;
		if( nCount%50 == 0 )
			printf("Loaded %ld img...,time:%.4f\n",nCount,run.time());
		
		cvReleaseImage(&img);img = 0;
	}

	/*********************************close file*************************************/
	if (fpListFile) {fclose(fpListFile);fpListFile = 0;}	

	/*********************************Release*************************************/
	api_muti_label.Release();

	/*********************************Print Info*********************************/
	if ( nCount != 0 ) 
	{
		printf( "nCount:%ld,nCountObj:%ld_%.4f,PredictTime:%.4fms\n", 
			nCount, nCountObj, nCountObj*1.0/nCount, allPredictTime*1000.0/nCount );
	}
	
	cout<<"Done!! "<<endl;
	
	return nRet;	
}

int main(int argc, char* argv[])
{
	int  ret = 0;
	char szKeyFiles[256],szSavePath[256];

	//inLabelClass:0-voc,1-coco,2-old in;
	if (argc == 8 && strcmp(argv[1],"-test") == 0) {
		ret = frcnn_test( argv[2], argv[3], argv[4], atof(argv[5]), atoi(argv[6]), atoi(argv[7]) );
	}
	else
	{
		cout << "usage:\n" << endl;
		cout << "\tDemo_mutilabel -test loadImagePath svPath keyfile MutiLabel_T binGPU deviceID\n" << endl;
		return ret;
	}
	return ret;
}

