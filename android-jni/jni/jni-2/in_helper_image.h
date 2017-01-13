#include <jni.h>
#include <android/bitmap.h>

//
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//
cv::Mat bitmap_to_mat(JNIEnv *env, jobject bmp) ;