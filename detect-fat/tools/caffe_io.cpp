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
	std::string deploy_file(argv[1]);
	std::string model_file(argv[2]);
	std::string net_file(argv[3]);

	//
	GeekeyeDLL ge;
	ge.init(deploy_file, model_file);

	//
	ge.save_net(net_file);

	return 0;
}