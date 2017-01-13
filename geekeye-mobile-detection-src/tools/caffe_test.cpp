#include <iostream>
#include <string>
#include <vector>


#include "geekeyedll/geekeyedll.h"
#include "common/common.h"
using namespace geekeyelab;

int main(int argc, char** argv)
{
	//
	if(argc < 3){
		std::cout<<"do <keyfile> <image file>."<<std::endl;
		return -1;
	}
	std::string keyfile(argv[1]);
	std::string image_file(argv[2]);
	std::string deploy_file(keyfile+"/SqueezeNet_v1.1/deploy.prototxt");
	std::string  model_file(keyfile+"/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel");
	//
	std::vector<float> mean_value(3, 0.0f);
	mean_value[0] = 104;
    mean_value[1] = 117;
    mean_value[2] = 123;
	GeekeyeDLL ge;
	ge.init(deploy_file, model_file, 0, 0);
	ge.set_mean_value(mean_value);

	//
	std::vector< std::pair<int, float> > results;
	geekeyelab::RunTimer<double> timer;  
  	timer.start();
  	ge.predict(image_file, results);
  	timer.end();
  	std::cout<<"predict time:"<<timer.time() << " s."<<std::endl; 
  	std::cout<<"index:"<<results[0].first<<" score:"<<results[0].second<<std::endl;

	//
	return 0;
}
