
#include "in_helper_image.h"

/*
*
*/
cv::Mat bitmap_to_mat(JNIEnv *env, jobject bmp) 
{
	//
	cv::Mat image;
	if (bmp == NULL)
		return image;

	//
	AndroidBitmapInfo info;
	int ret;
	if ((ret = AndroidBitmap_getInfo(env, bmp, &info)) < 0) {
		return image;
	}
	if (info.width <= 0 || info.height <= 0
			|| (info.format != ANDROID_BITMAP_FORMAT_RGB_565
					&& info.format != ANDROID_BITMAP_FORMAT_RGBA_8888
					&& info.format != ANDROID_BITMAP_FORMAT_RGBA_4444)) {
		return image;
	}

	// 
	int width = info.width;
    int height = info.height;
    int widthStep = width;

	// Lock the bitmap to get the buffer
	void * pixels = NULL;
	ret = AndroidBitmap_lockPixels(env, bmp, &pixels);
	//unsigned char* udata = (unsigned char*) pixels;
	//ret = AndroidBitmap_lockPixels(env, bmp, (void**)&udata);
	if (ret < 0 || pixels == NULL)
		return image;


	//
	{
		switch (info.format) {
		case ANDROID_BITMAP_FORMAT_RGB_565: {

			break;
		}
		case ANDROID_BITMAP_FORMAT_RGBA_8888: {
			cv::Mat image_RGBA = cv::Mat(height, width, CV_8UC4, (unsigned char*) pixels);
			cvtColor(image_RGBA, image, CV_RGBA2BGR);	//RGBA2BGR
			//cvtColor(image_RGBA,image,CV_BGR2RGBA);	//BGR2RGBA

			break;
		}
		case ANDROID_BITMAP_FORMAT_RGBA_4444:

			break;
		} //switch
	}

	//
	AndroidBitmap_unlockPixels(env, bmp);
	return image;
}