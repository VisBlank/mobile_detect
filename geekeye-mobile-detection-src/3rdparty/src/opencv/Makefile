##
include Makefile.cv

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



### sources


### flags
#include && lib
INCLUDE += $(CV_INC)



#flags
CFLAGS =    $(INCLUDE) -O3 -Os -pipe  -fvisibility=hidden 
CFLAGS +=  -DCPU_ONLY  -DHAVE_PTHREAD     -DHAVE_JPEG -DHAVE_PNG
CXXFLAGS =  $(INCLUDE)   -std=c++11 -O3 -Os -pipe  -fvisibility=hidden 
CXXFLAGS +=  -DCPU_ONLY  -DHAVE_PTHREAD    -DHAVE_JPEG -DHAVE_PNG

#
LIBARAY += -L/usr/local/lib

#
LDFLAGS  += -lm -lc -ldl   -lz 
LDFLAGS  += -lpthread #-ljpeg -lpng


#
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

all: $(OBJECTS) 
	@echo ---------------------------------------
	$(CXX) $(CXXFLAGS) -shared  -fPIC  -o $(LIB)/libincv.so $(OBJECTS) $(LIBRARY) $(LDFLAGS)
	@echo ---------------------------------------




clean:
	@echo $(OBJECTS)
	#rm -f $(OBJECTS) 

