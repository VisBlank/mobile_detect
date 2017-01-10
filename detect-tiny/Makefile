##
#include 3rdparty/src/opencv/Makefile.cv

###compiler
CXX :=g++ -fPIC  -ggdb 
CC :=gcc   -ggdb 

#shared library
#SFLAG := -fPIC -shared

#
BIN=bin
LIB=lib



### sources  
#  
CAFFE_DIR = src/caffe
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

#
API_MUTILABEL_DIR = ./src/API_mutilabel
API_MUTILABEL_SRC = $(API_MUTILABEL_DIR)/API_caffe_pvanet.cpp $(API_MUTILABEL_DIR)/API_mutilabel_pvanet.cpp

### flags
#include && lib
INCLUDE += -I. -I./include -I./src
INCLUDE += $(CAFFE_INC)
#INCLUDE += $(CV_INC)
INCLUDE += -I/usr/local/include/opencv #-I/usr/local/include 
INCLUDE += -I/opt/OpenBLAS/include


#flags
CFLAGS =    $(INCLUDE) -O3 -Os -pipe 
CFLAGS +=  -DCPU_ONLY  -DHAVE_PTHREAD -DBLAS_OPEN   -DHAVE_JPEG -DHAVE_PNG 
CXXFLAGS =  $(INCLUDE)   -std=c++11 -O3 -Os -pipe 
CXXFLAGS +=  -DCPU_ONLY  -DHAVE_PTHREAD -DBLAS_OPEN   -DHAVE_JPEG -DHAVE_PNG 

#
LIBARAY += -L/usr/local/lib
LIBARAY += -L/opt/OpenBLAS/lib

#
LDFLAGS  += -lm -lc -ldl   -lz 
LDFLAGS  += -lpthread -ljpeg -lpng
#LDFLAGS  += -lprotobuf 
LDFLAGS  += -lopenblas
LDFLAGS  += -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio


#
SRCS_CPP += src/geekeyedll/geekeye_dll.cpp  
SRCS_CPP += $(CAFFE_SRCS)
SRCS_CPP += $(API_MUTILABEL_SRC)
#SRCS_CPP += $(PROTO_SRCS)
#SRCS_CPP += $(CV_SRCS)
#OBJS_CPP = $(SRCS_CPP:%.cpp=%.o)
#OBJS_C = $(SRCS_C:%.c=%.o)
OBJECTS_CPP = $(SRCS_CPP:%.cpp=%.o)
OBJECTS_C = $(OBJECTS_CPP:%.c=%.o)
OBJECTS = $(OBJECTS_C:%.cc=%.o)


###### generation rules
#.SUFFIXES: .cpp .c .o .so .a .d
.c.o:
	@echo ---------------------------------------
	$(CC) $(CFLAGS) -c $< -o $@
	@echo ---------------------------------------

.cpp.o:
	@echo ---------------------------------------
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo ---------------------------------------


###### main
all: demo

demo: $(OBJECTS) 
	@echo ---------------------------------------
	mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) tools/demo.cpp -o $(BIN)/demo $(OBJECTS) $(LIBARAY) $(LDFLAGS)
	@echo ---------------------------------------



clean:
	@echo $(OBJECTS)
	rm -f $(OBJECTS) 
	rm $(BIN)/*


