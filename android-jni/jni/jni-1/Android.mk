#LOCAL_ALLOW_UNDEFINED_SYMBOLS := true
#
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
include $(LOCAL_PATH)/geekeye-mobile-detection-src/3rdparty/src/opencv/Makefile.cv

#
LOCAL_MODULE    := Geekeye_Mobile_Detection_1_0_0

### include
API_PATH = $(LOCAL_PATH)/geekeye-mobile-detection-src
PARTY_INC = $(API_PATH)/3rdparty/include
LOCAL_C_INCLUDES += $(PARTY_INC) $(API_PATH)/include ./ $(CV_INC)

### sources
CAFFE_DIR = $(API_PATH)/3rdparty/src/caffe
CAFFE_LAYER_DIR = $(CAFFE_DIR)/layers
CAFFE_LAYER_SRCS_1 = $(CAFFE_LAYER_DIR)/conv_layer.cpp $(CAFFE_LAYER_DIR)/base_conv_layer.cpp $(CAFFE_LAYER_DIR)/lrn_layer.cpp \
        	$(CAFFE_LAYER_DIR)/eltwise_layer.cpp $(CAFFE_LAYER_DIR)/pooling_layer.cpp $(CAFFE_LAYER_DIR)/neuron_layer.cpp \
        	$(CAFFE_LAYER_DIR)/power_layer.cpp $(CAFFE_LAYER_DIR)/split_layer.cpp $(CAFFE_LAYER_DIR)/relu_layer.cpp \
        	$(CAFFE_LAYER_DIR)/sigmoid_layer.cpp $(CAFFE_LAYER_DIR)/softmax_layer.cpp $(CAFFE_LAYER_DIR)/tanh_layer.cpp \
        	$(CAFFE_LAYER_DIR)/flatten_layer.cpp $(CAFFE_LAYER_DIR)/concat_layer.cpp $(CAFFE_LAYER_DIR)/spp_layer.cpp
CAFFE_LAYER_SRCS_2 = $(CAFFE_LAYER_DIR)/dropout_layer.cpp $(CAFFE_LAYER_DIR)/inner_product_layer.cpp \
	        $(CAFFE_LAYER_DIR)/batch_norm_layer.cpp $(CAFFE_LAYER_DIR)/scale_layer.cpp $(CAFFE_LAYER_DIR)/bias_layer.cpp\
			$(CAFFE_LAYER_DIR)/absval_layer.cpp $(CAFFE_LAYER_DIR)/deconv_layer.cpp $(CAFFE_LAYER_DIR)/dummy_data_layer.cpp \
			$(CAFFE_LAYER_DIR)/loss_layer.cpp $(CAFFE_LAYER_DIR)/proposal_layer.cpp $(CAFFE_LAYER_DIR)/reshape_layer.cpp \
			$(CAFFE_LAYER_DIR)/roi_pooling_layer.cpp $(CAFFE_LAYER_DIR)/smooth_L1_loss_layer.cpp
CAFFE_SRCS = $(CAFFE_DIR)/common.cpp $(CAFFE_DIR)/syncedmem.cpp $(CAFFE_DIR)/blob.cpp $(CAFFE_DIR)/layer.cpp \
        	$(CAFFE_DIR)/layer_factory.cpp $(CAFFE_DIR)/net.cpp $(CAFFE_DIR)/proto/caffe.pb.cpp \
	        $(CAFFE_DIR)/util/insert_splits.cpp $(CAFFE_DIR)/util/io.cpp $(CAFFE_DIR)/util/upgrade_proto.cpp \
	        $(CAFFE_DIR)/util/math_functions.cpp $(CAFFE_DIR)/util/im2col.cpp $(CAFFE_DIR)/util/nms.cpp
API_SRCS =  $(API_PATH)/src/API_mutilabel/API_caffe_pvanet.cpp \
			$(API_PATH)/src/API_mutilabel/API_mutilabel_pvanet.cpp
JNI_SRCS = 	in_helper_image.cpp com_example_jni_geekeye_mobile_detection_1_0_0.cpp
LOCAL_SRC_FILES := $(CAFFE_LAYER_SRCS_1) 
LOCAL_SRC_FILES += $(CAFFE_LAYER_SRCS_2)
LOCAL_SRC_FILES += $(CAFFE_SRCS)
LOCAL_SRC_FILES += $(CV_SRCS)
LOCAL_SRC_FILES += $(API_SRCS)
LOCAL_SRC_FILES += $(JNI_SRCS)

### flags
LOCAL_CFLAGS   +=  -I. -I$(API_PATH)/3rdparty/include
LOCAL_CFLAGS   +=  -DHAVE_JPEG -DHAVE_PNG -O3 -Os -pipe -fvisibility=hidden  -s -Wl,-x,--exclude-libs=ALL -fexceptions
LOCAL_CFLAGS += -DCPU_ONLY -DHAVE_PTHREAD  -DBLAS_OPEN  #-DANDROID_LOG
#
LOCAL_CPPFLAGS +=  -I. -I$(API_PATH)/3rdparty/include
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
LOCAL_WHOLE_STATIC_LIBRARIES += protobuf_static

include $(BUILD_SHARED_LIBRARY)

$(call import-add-path, /cygdrive/f/coding/geekeye-mobile-detection/jni/geekeye-mobile-detection-src/3rdparty/libs/android)
$(call import-module, libopenblas)
$(call import-module, libprotobuf)

