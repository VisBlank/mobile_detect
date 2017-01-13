#include <iostream>
#include <string>
#include <vector>


#include "caffe/caffe.hpp"
#include "common/common.h"

int main(int argc, char** argv)
{
	//
	if(argc < 2){
		std::cout<<"do <model file>."<<std::endl;
		return -1;
	}
	std::string model_file(argv[1]);
	//
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	caffe::Net<float> caffe_net(model_file, caffe::TEST); 
	//
	geekeyelab::RunTimer<double> timer;  
  	int iters = 50;
  	float iter_loss;
  	double time_tol = 0.0;
  	double time_iter = 0.0;
  	for(int i = 0; i<iters; i++){
  		timer.start();
  		caffe_net.ForwardPrefilled(&iter_loss);
  		timer.end();
  		time_iter = timer.time();
  		std::cout<<i<<"-th forward time:"<<time_iter<<" s"<<std::endl;
  		time_tol += time_iter;
  	}
  	
  	std::cout<<"model: "<<model_file<<std::endl;
  	std::cout<<"forward time avg: "<<time_tol/iters<<" s"<<std::endl;
	//
	return 0;
}
