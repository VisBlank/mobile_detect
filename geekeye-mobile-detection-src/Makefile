##
include 3rdparty/src/opencv/Makefile.cv

###compiler
CXX :=g++ -fPIC  -ggdb 
CC :=gcc   -ggdb 

#shared library
#SFLAG := -fPIC -shared

#
BIN=bin
LIB=lib

### include
#
PARTY_INC = -I./3rdparty/include 
CAFFE_INC = -I./3rdparty/include/caffe

### sources
CAFFE_DIR = ./3rdparty/src/caffe
CAFFE_SRCS = $(CAFFE_DIR)/util/math_functions.cpp $(CAFFE_DIR)/common.cpp $(CAFFE_DIR)/syncedmem.cpp   \
        $(CAFFE_DIR)/proto/caffe.pb.cpp $(CAFFE_DIR)/util/im2col.cpp  $(CAFFE_DIR)/blob.cpp \
        $(CAFFE_DIR)/layer.cpp $(CAFFE_DIR)/layer_factory.cpp \
        $(CAFFE_DIR)/layers/conv_layer.cpp $(CAFFE_DIR)/layers/base_conv_layer.cpp $(CAFFE_DIR)/layers/lrn_layer.cpp \
        $(CAFFE_DIR)/layers/eltwise_layer.cpp $(CAFFE_DIR)/layers/pooling_layer.cpp $(CAFFE_DIR)/layers/neuron_layer.cpp \
        $(CAFFE_DIR)/layers/power_layer.cpp $(CAFFE_DIR)/layers/split_layer.cpp $(CAFFE_DIR)/layers/relu_layer.cpp \
        $(CAFFE_DIR)/layers/sigmoid_layer.cpp $(CAFFE_DIR)/layers/softmax_layer.cpp $(CAFFE_DIR)/layers/tanh_layer.cpp \
        $(CAFFE_DIR)/layers/flatten_layer.cpp $(CAFFE_DIR)/layers/concat_layer.cpp $(CAFFE_DIR)/layers/spp_layer.cpp \
        $(CAFFE_DIR)/layers/dropout_layer.cpp $(CAFFE_DIR)/layers/inner_product_layer.cpp \
        $(CAFFE_DIR)/layers/batch_norm_layer.cpp  $(CAFFE_DIR)/layers/scale_layer.cpp $(CAFFE_DIR)/layers/bias_layer.cpp\
	$(CAFFE_DIR)/layers/absval_layer.cpp $(CAFFE_DIR)/layers/deconv_layer.cpp $(CAFFE_DIR)/layers/dummy_data_layer.cpp \
	$(CAFFE_DIR)/layers/loss_layer.cpp $(CAFFE_DIR)/layers/proposal_layer.cpp $(CAFFE_DIR)/layers/reshape_layer.cpp \
	$(CAFFE_DIR)/layers/roi_pooling_layer.cpp $(CAFFE_DIR)/layers/smooth_L1_loss_layer.cpp \
        $(CAFFE_DIR)/util/insert_splits.cpp $(CAFFE_DIR)/util/io.cpp $(CAFFE_DIR)/util/upgrade_proto.cpp \
	$(CAFFE_DIR)/util/nms.cpp \
	$(CAFFE_DIR)/net.cpp 

#
GEEKEYEDLL_DIR = ./src/geekeyedll
GEEKEYEDLL_SRC = $(GEEKEYEDLL_DIR)/geekeyedll.cpp 

#
API_MUTILABEL_DIR = ./src/API_mutilabel
API_MUTILABEL_SRC = $(API_MUTILABEL_DIR)/API_caffe_pvanet.cpp $(API_MUTILABEL_DIR)/API_mutilabel_pvanet.cpp

### flags
#include && lib
INCLUDE += -I./include
INCLUDE += $(PARTY_INC) 
INCLUDE += $(CAFFE_INC)
INCLUDE += $(CV_INC)
INCLUDE += -I/usr/local/include -I/usr/include
INCLUDE += -I/opt/OpenBLAS/include

#flags
CFLAGS =    $(INCLUDE) -O3 -Os -pipe 
CFLAGS +=  -DCPU_ONLY  -DHAVE_PTHREAD -DBLAS_OPEN   -DHAVE_JPEG -DHAVE_PNG
CXXFLAGS =  $(INCLUDE)   -std=c++11 -O3 -Os -pipe 
CXXFLAGS +=  -DCPU_ONLY  -DHAVE_PTHREAD -DBLAS_OPEN   -DHAVE_JPEG -DHAVE_PNG

#
LIBARAY += -L/usr/local/lib -L/usr/lib
LIBARAY += -L/opt/OpenBLAS/lib

#
LDFLAGS  += -lm -lc -ldl   -lz 
LDFLAGS  += -lpthread -ljpeg -lpng
LDFLAGS  += -lprotobuf
LDFLAGS  += -lopenblas
#LDFLAGS  += -lglog -lboost_system -lrt -lcurl -lstdc++

#
SRCS_CPP += $(CAFFE_SRCS)
SRCS_CPP += $(GEEKEYEDLL_SRC)
SRCS_CPP += $(API_MUTILABEL_SRC)
SRCS_CPP += $(CV_SRCS)
#OBJS_CPP = $(SRCS_CPP:%.cpp=%.o)
#OBJS_C = $(SRCS_C:%.c=%.o)
OBJECTS_CPP = $(SRCS_CPP:%.cpp=%.o)
OBJECTS = $(OBJECTS_CPP:%.c=%.o)


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
all: caffe_time  caffe_test loc_test Demo_mutilabel_pvanet

caffe_time: $(OBJECTS) 
	@echo ---------------------------------------
	$(CXX) $(CXXFLAGS) tools/caffe_time.cpp -o $(BIN)/caffe_time $(OBJECTS) $(LIBARAY) $(LDFLAGS)
	@echo ---------------------------------------

caffe_test: $(OBJECTS) 
	@echo ---------------------------------------
	$(CXX) $(CXXFLAGS) tools/caffe_test.cpp -o $(BIN)/caffe_test $(OBJECTS) $(LIBARAY) $(LDFLAGS)
	@echo ---------------------------------------

loc_test: $(OBJECTS) 
	@echo ---------------------------------------
	$(CXX) $(CXXFLAGS) tools/loc_test.cpp -o $(BIN)/loc_test $(OBJECTS) $(LIBARAY) $(LDFLAGS)
	@echo ---

Demo_mutilabel_pvanet: $(OBJECTS) 
	@echo ---------------------------------------
	$(CXX) $(CXXFLAGS) tools/Demo_mutilabel_pvanet.cpp -o $(BIN)/Demo_mutilabel_pvanet $(OBJECTS) $(LIBARAY) $(LDFLAGS)
	@echo ---------------------------------------

clean:
	@echo $(OBJECTS)
	rm -f $(OBJECTS) 

