
#include "in_helper_jni.h"

/*
 * Class:     
 * Method:    
 * Signature: 
 */
jstring char2jstring(JNIEnv* pEnv, const char* pChars, int Length,
		const char* szCharset) 
{
	jclass clsString = pEnv->FindClass("java/lang/String");
	jmethodID construct = pEnv->GetMethodID(clsString, "<init>",
			"([BLjava/lang/String;)V");
	jbyteArray byteArray = pEnv->NewByteArray(Length);
	pEnv->SetByteArrayRegion(byteArray, 0, Length, (jbyte*) pChars);
	jstring charset = pEnv->NewStringUTF(szCharset);
	jstring strDst = (jstring) pEnv->NewObject(clsString, construct, byteArray,
			charset);
	pEnv->DeleteLocalRef(byteArray);
	return strDst;
}


/*
 * Class:     
 * Method:    
 * Signature: 
 */
int jstring2char(JNIEnv* pEnv, char** ppChars, int* pLength, jstring jstr,
		const char* szCharset) 
{
	jclass clsString = pEnv->FindClass("java/lang/String");
	jstring charset = pEnv->NewStringUTF(szCharset);
	jmethodID method = pEnv->GetMethodID(clsString, "getBytes",
			"(Ljava/lang/String;)[B");
	jbyteArray byteArray = (jbyteArray) pEnv->CallObjectMethod(jstr, method,
			charset);
	jsize nLength = pEnv->GetArrayLength(byteArray);
	jbyte* bytes = pEnv->GetByteArrayElements(byteArray, JNI_FALSE);
	*pLength = nLength;
	*ppChars = (char*) malloc(nLength);
	memcpy(*ppChars, bytes, nLength);
	pEnv->ReleaseByteArrayElements(byteArray, bytes, 0);
	pEnv->DeleteLocalRef(charset);
	return 1;
}