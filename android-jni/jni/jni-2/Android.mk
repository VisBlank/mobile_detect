#LOCAL_ALLOW_UNDEFINED_SYMBOLS := true
#
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

#
LOCAL_MODULE := Geekeye_Mobile_Detection_1_0_0

### include
API_PATH = $(LOCAL_PATH)/detect-tiny
INCLUDE_PATH = $(API_PATH)/include
SRC_PATH = $(API_PATH)/src
LOCAL_C_INCLUDES += $(INCLUDE_PATH) ./

### sources
CAFFE_DIR = $(SRC_PATH)/caffe
CAFFE_SRCS = $(CAFFE_DIR)/util/math_functions.cpp $(CAFFE_DIR)/common.cpp $(CAFFE_DIR)/syncedmem.cpp   \
        $(CAFFE_DIR)/util/im2col.cpp  $(CAFFE_DIR)/blob.cpp $(CAFFE_DIR)/layer.cpp  \
        $(CAFFE_DIR)/layers/conv_layer.cpp $(CAFFE_DIR)/layers/base_conv_layer.cpp  \
        $(CAFFE_DIR)/layers/pooling_layer.cpp $(CAFFE_DIR)/layers/neuron_layer.cpp \
        $(CAFFE_DIR)/layers/power_layer.cpp $(CAFFE_DIR)/layers/split_layer.cpp $(CAFFE_DIR)/layers/relu_layer.cpp \
        $(CAFFE_DIR)/layers/sigmoid_layer.cpp $(CAFFE_DIR)/layers/softmax_layer.cpp $(CAFFE_DIR)/layers/tanh_layer.cpp \
        $(CAFFE_DIR)/layers/concat_layer.cpp $(CAFFE_DIR)/layers/dropout_layer.cpp  \
		$(CAFFE_DIR)/util/nms.cpp $(CAFFE_DIR)/layers/reshape_layer.cpp \
		$(CAFFE_DIR)/layers/proposal_layer.cpp $(CAFFE_DIR)/layers/roi_pooling_layer.cpp  \
		$(CAFFE_DIR)/layers/deconv_layer.cpp $(CAFFE_DIR)/layers/inner_product_layer.cpp \
        $(CAFFE_DIR)/net.cpp
API_SRCS =  $(SRC_PATH)/API_mutilabel/API_caffe_pvanet.cpp \
			$(SRC_PATH)/API_mutilabel/API_mutilabel_pvanet.cpp
JNI_SRCS = com_example_jni_geekeye_mobile_detection_1_0_0.cpp
#JNI_SRCS = in_helper_image.cpp 
LOCAL_SRC_FILES += $(CAFFE_SRCS)
LOCAL_SRC_FILES += $(API_SRCS)
LOCAL_SRC_FILES += $(JNI_SRCS)

### flags
LOCAL_CFLAGS   +=  -I. -I$(INCLUDE_PATH)
LOCAL_CFLAGS   +=  -DHAVE_JPEG -DHAVE_PNG -O3 -Os -pipe -fvisibility=hidden  -s -Wl,-x,--exclude-libs=ALL -fexceptions
LOCAL_CFLAGS += -DCPU_ONLY -DHAVE_PTHREAD  -DBLAS_OPEN  #-DANDROID_LOG
#
LOCAL_CPPFLAGS +=  -I. -I$(INCLUDE_PATH)
LOCAL_CPPFLAGS +=  -DHAVE_JPEG -DHAVE_PNG  -O3 -Os -pipe -fvisibility=hidden  -std=c++11 -s -Wl,-x,--exclude-libs=ALL -frtti
LOCAL_CPPFLAGS += -DCPU_ONLY -DHAVE_PTHREAD -DBLAS_OPEN  #-DANDROID_LOG

### libs
LOCAL_LDLIBS += -lm -ldl -lz -lc -llog -ljnigraphics

#
TARGET_CFLAGS +=  -D_NDK_MATH_NO_SOFTFP=1
TARGET_LDFLAGS += -Wl,--no-warn-mismatch
ifeq ($(TARGET_ARCH_ABI), armeabi-v7a-hard)
$(warning $(TARGET_ARCH_ABI))
TARGET_CFLAGS += -mfloat-abi=hard -mhard-float -mfpu=vfp
TARGET_LDFLAGS += -lm_hard
endif
ifeq ($(TARGET_ARCH_ABI), arm64-v8a)
$(warning $(TARGET_ARCH_ABI))
#TARGET_CFLAGS += -mfloat-abi=hard -mhard-float -mfpu=vfp
#TARGET_LDFLAGS += -lm_hard
endif

#
LOCAL_STATIC_LIBRARIES += -L/cygdrive/e/android-ndk-r10b/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(TARGET_ARCH_ABI) -lgnustl_static
LOCAL_C_INCLUDES += /cygdrive/e/android-ndk-r10b/sources/cxx-stl/gnu-libstdc++/4.9/include
LOCAL_CFLAGS += -I/cygdrive/e/android-ndk-r10b/sources/cxx-stl/gnu-libstdc++/4.9/include

#NDK_MODULE_PATH += $(API_PATH)/3rdparty
LOCAL_WHOLE_STATIC_LIBRARIES += libopenblas-$(TARGET_ARCH_ABI)
#LOCAL_WHOLE_STATIC_LIBRARIES += protobuf_static

include $(BUILD_SHARED_LIBRARY)

$(call import-add-path, /cygdrive/f/coding/geekeye-mobile-detection/jni/detect-tiny/libs/android)
$(call import-module, libopenblas)
#$(call import-module, libprotobuf)

