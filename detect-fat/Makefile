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
PROTO_DIR = src/google
PROTO_SRCS =   $(PROTO_DIR)/protobuf/stubs/bytestream.cc \
        $(PROTO_DIR)/protobuf/stubs/common.cc $(PROTO_DIR)/protobuf/stubs/int128.cc $(PROTO_DIR)/protobuf/stubs/once.cc $(PROTO_DIR)/protobuf/stubs/status.cc $(PROTO_DIR)/protobuf/stubs/statusor.cc \
        $(PROTO_DIR)/protobuf/stubs/stringpiece.cc $(PROTO_DIR)/protobuf/stubs/stringprintf.cc $(PROTO_DIR)/protobuf/stubs/structurally_valid.cc $(PROTO_DIR)/protobuf/stubs/strutil.cc \
           $(PROTO_DIR)/protobuf/arena.cc $(PROTO_DIR)/protobuf/arenastring.cc $(PROTO_DIR)/protobuf/extension_set.cc $(PROTO_DIR)/protobuf/generated_message_util.cc \
        $(PROTO_DIR)/protobuf/message_lite.cc $(PROTO_DIR)/protobuf/repeated_field.cc $(PROTO_DIR)/protobuf/wire_format_lite.cc $(PROTO_DIR)/protobuf/io/coded_stream.cc $(PROTO_DIR)/protobuf/io/zero_copy_stream.cc \
        $(PROTO_DIR)/protobuf/io/zero_copy_stream_impl_lite.cc $(PROTO_DIR)/protobuf/any.pb.cc $(PROTO_DIR)/protobuf/api.pb.cc $(PROTO_DIR)/protobuf/stubs/mathlimits.cc $(PROTO_DIR)/protobuf/any.cc \
        $(PROTO_DIR)/protobuf/descriptor.cc $(PROTO_DIR)/protobuf/descriptor_database.cc $(PROTO_DIR)/protobuf/descriptor.pb.cc $(PROTO_DIR)/protobuf/duration.pb.cc $(PROTO_DIR)/protobuf/dynamic_message.cc \
        $(PROTO_DIR)/protobuf/empty.pb.cc $(PROTO_DIR)/protobuf/extension_set_heavy.cc $(PROTO_DIR)/protobuf/field_mask.pb.cc $(PROTO_DIR)/protobuf/generated_message_reflection.cc $(PROTO_DIR)/protobuf/map_field.cc \
        $(PROTO_DIR)/protobuf/message.cc $(PROTO_DIR)/protobuf/reflection_ops.cc $(PROTO_DIR)/protobuf/service.cc $(PROTO_DIR)/protobuf/source_context.pb.cc $(PROTO_DIR)/protobuf/struct.pb.cc $(PROTO_DIR)/protobuf/stubs/substitute.cc \
        $(PROTO_DIR)/protobuf/text_format.cc  $(PROTO_DIR)/protobuf/type.pb.cc $(PROTO_DIR)/protobuf/unknown_field_set.cc $(PROTO_DIR)/protobuf/wire_format.cc $(PROTO_DIR)/protobuf/wrappers.pb.cc \
        $(PROTO_DIR)/protobuf/io/gzip_stream.cc $(PROTO_DIR)/protobuf/io/printer.cc $(PROTO_DIR)/protobuf/io/strtod.cc $(PROTO_DIR)/protobuf/io/tokenizer.cc $(PROTO_DIR)/protobuf/io/zero_copy_stream_impl.cc $(PROTO_DIR)/protobuf/compiler/importer.cc \
        $(PROTO_DIR)/protobuf/compiler/parser.cc $(PROTO_DIR)/protobuf/util/field_comparator.cc $(PROTO_DIR)/protobuf/util/field_mask_util.cc $(PROTO_DIR)/protobuf/util/internal/datapiece.cc $(PROTO_DIR)/protobuf/util/internal/default_value_objectwriter.cc \
        $(PROTO_DIR)/protobuf/util/internal/error_listener.cc $(PROTO_DIR)/protobuf/util/internal/field_mask_utility.cc $(PROTO_DIR)/protobuf/util/internal/json_escaping.cc $(PROTO_DIR)/protobuf/util/internal/json_objectwriter.cc \
        $(PROTO_DIR)/protobuf/util/internal/json_stream_parser.cc $(PROTO_DIR)/protobuf/util/internal/object_writer.cc $(PROTO_DIR)/protobuf/util/internal/protostream_objectsource.cc $(PROTO_DIR)/protobuf/util/internal/protostream_objectwriter.cc \
        $(PROTO_DIR)/protobuf/util/internal/proto_writer.cc $(PROTO_DIR)/protobuf/util/internal/type_info.cc $(PROTO_DIR)/protobuf/util/internal/type_info_test_helper.cc $(PROTO_DIR)/protobuf/util/internal/utility.cc $(PROTO_DIR)/protobuf/util/json_util.cc \
        $(PROTO_DIR)/protobuf/util/message_differencer.cc  $(PROTO_DIR)/protobuf/util/type_resolver_util.cc \
        $(PROTO_DIR)/protobuf/util/time_util.cc   $(PROTO_DIR)/protobuf/timestamp.pb.cc $(PROTO_DIR)/protobuf/stubs/time.cc

CAFFE_DIR = src/caffe
CAFFE_SRCS = $(CAFFE_DIR)/util/math_functions.cpp $(CAFFE_DIR)/common.cpp $(CAFFE_DIR)/syncedmem.cpp   \
        $(CAFFE_DIR)/proto/caffe.pb.cpp $(CAFFE_DIR)/util/im2col.cpp  $(CAFFE_DIR)/blob.cpp \
        $(CAFFE_DIR)/layer.cpp $(CAFFE_DIR)/layer_factory.cpp \
        $(CAFFE_DIR)/layers/conv_layer.cpp $(CAFFE_DIR)/layers/base_conv_layer.cpp $(CAFFE_DIR)/layers/lrn_layer.cpp \
        $(CAFFE_DIR)/layers/eltwise_layer.cpp $(CAFFE_DIR)/layers/pooling_layer.cpp $(CAFFE_DIR)/layers/neuron_layer.cpp \
        $(CAFFE_DIR)/layers/power_layer.cpp $(CAFFE_DIR)/layers/split_layer.cpp $(CAFFE_DIR)/layers/relu_layer.cpp \
        $(CAFFE_DIR)/layers/sigmoid_layer.cpp $(CAFFE_DIR)/layers/softmax_layer.cpp $(CAFFE_DIR)/layers/tanh_layer.cpp \
        $(CAFFE_DIR)/layers/flatten_layer.cpp $(CAFFE_DIR)/layers/concat_layer.cpp $(CAFFE_DIR)/layers/spp_layer.cpp \
        $(CAFFE_DIR)/layers/dropout_layer.cpp $(CAFFE_DIR)/layers/inner_product_layer.cpp \
        $(CAFFE_DIR)//layers/batch_norm_layer.cpp  $(CAFFE_DIR)/layers/scale_layer.cpp $(CAFFE_DIR)/layers/bias_layer.cpp\
        $(CAFFE_DIR)/layers/prelu_layer.cpp  $(CAFFE_DIR)/layers/slice_layer.cpp \
        $(CAFFE_DIR)/util/insert_splits.cpp $(CAFFE_DIR)/util/io.cpp $(CAFFE_DIR)/util/upgrade_proto.cpp \
	$(CAFFE_DIR)/layers/deconv_layer.cpp $(CAFFE_DIR)/layers/reshape_layer.cpp \
	$(CAFFE_DIR)/layers/proposal_layer.cpp $(CAFFE_DIR)/layers/roi_pooling_layer.cpp $(CAFFE_DIR)/util/nms.cpp \
	$(CAFFE_DIR)/net.cpp 

#
API_MUTILABEL_DIR = ./src/API_mutilabel
API_MUTILABEL_SRC = $(API_MUTILABEL_DIR)/API_caffe_pvanet.cpp $(API_MUTILABEL_DIR)/API_mutilabel_pvanet.cpp

### flags
#include && lib
INCLUDE += -I. -I./include 
#INCLUDE += -I./src
INCLUDE += $(CAFFE_INC)
#INCLUDE += $(CV_INC)
INCLUDE += -I/usr/local/include/opencv
INCLUDE += -I/opt/OpenBLAS/include
INCLUDE += -I/usr/local/include -I/usr/include


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

