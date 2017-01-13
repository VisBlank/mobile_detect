
#include <jni.h>

/*
 * Class:     
 * Method:    
 * Signature: 
 */
jstring char2jstring(JNIEnv* pEnv, const char* pChars, int Length,
		const char* szCharset);


/*
 * Class:     
 * Method:    
 * Signature: 
 */
int jstring2char(JNIEnv* pEnv, char** ppChars, int* pLength, jstring jstr,
		const char* szCharset);