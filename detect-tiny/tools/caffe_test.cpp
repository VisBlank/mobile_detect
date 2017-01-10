#include <iostream>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "geekeyedll/geekeye_dll.h"
#include "common/common.h"
using namespace geekeyelab;

int main(int argc, char** argv)
{
	//
	std::string model_file(argv[1]);

	//
	GeekeyeDLL ge;
	ge.init(model_file, 0, 0);
	std::vector<float> mean_value(3, 0.0f);
	mean_value[0] = 104;
    mean_value[1] = 117;
    mean_value[2] = 123;
    ge.set_mean_value(mean_value);
    
	//
	std::string image_file(argv[2]);
	std::vector< std::pair<int, float> > results;
	ge.predict(image_file, results);
	std::cout<<results[0].first<<" "<<results[0].second<<std::endl;

	return 0;
}