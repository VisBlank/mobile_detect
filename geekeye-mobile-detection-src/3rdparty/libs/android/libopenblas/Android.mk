LOCAL_PATH:= $(call my-dir)
#libopenblas_armv7p-r0.2.16.dev.a
#libopenblas_armv8p-r0.2.16.dev.a 
#arm64-v8a	armeabi		armeabi-v7a	mips		mips64		x86		x86_64

include $(CLEAR_VARS)
LOCAL_MODULE:= libopenblas-armeabi
LOCAL_SRC_FILES:= libs/armeabi/libopenblas_armv6p-r0.2.19.dev.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE:= libopenblas-armeabi-v7a
LOCAL_SRC_FILES:= libs/armeabi-v7a/libopenblas_armv7p-r0.2.19.dev.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE:= libopenblas-armeabi-v7a-hard
LOCAL_SRC_FILES:= libs/armeabi-v7a/libopenblas_armv7p-r0.2.19.dev.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE:= libopenblas-arm64-v8a
LOCAL_SRC_FILES:= libs/arm64-v8a/libopenblas_armv8p-r0.2.19.dev.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE:= libopenblas-x86
LOCAL_SRC_FILES:= libs/x86/libopenblas_nehalemp-r0.2.19.dev.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE:= libopenblas-x86_64
LOCAL_SRC_FILES:= libs/x86_64/libopenblas_haswellp-r0.2.19.dev.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(PREBUILT_STATIC_LIBRARY)


