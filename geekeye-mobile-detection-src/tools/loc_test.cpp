#include <iostream>
#include <string>
#include <vector>


#include "geekeyedll/geekeyedll.h"
#include "common/common.h"
using namespace geekeyelab;


int main(int argc, char** argv)
{
	//
	if(argc < 2){
		std::cout<<"do  <image file>."<<std::endl;
		return -1;
	}
	std::string image_file(argv[1]);
	std::string deploy_file("models/SqueezeNet_v1.1/sz_apple_cat_dog_deploy.prototxt");
    std::string  model_file("models/SqueezeNet_v1.1/sz_apple_cat_dog.caffemodel");
    //
	std::vector<float> mean_value(3, 0.0f);
	mean_value[0] = 104;
    mean_value[1] = 117;
    mean_value[2] = 123;
	GeekeyeDLL ge;
	ge.init(deploy_file, model_file, 0, 0);
	ge.set_mean_value(mean_value);

	//
	cv::Mat image = cv::imread(image_file, 1);
	IplImage tmp = image;
	//
	std::string weight_layer("conv10_ft");
	std::string activity_layer("fire9/concat");
	cv::Mat heatmap;
	std::vector< std::pair<int, float> > results;
	std::vector<cv::Rect> rects;
	geekeyelab::RunTimer<double> timer;  
  	timer.start();
  	ge.predict_location(&tmp, results, weight_layer, activity_layer, rects, heatmap);
  	timer.end();
  	std::cout<<"predict time:"<<timer.time() << " s."<<std::endl; 
  	std::cout<<"index:"<<results[0].first<<" score:"<<results[0].second<<std::endl;
  	//
  	cv::imwrite("heatmap.jpg", heatmap);
  	for(int i = 0; i<rects.size(); i++){
  		cv::Rect r = rects[i];
  		
  		cv::rectangle(image, r,cv::Scalar(0,0,255),2); 
  	}
  	cv::imwrite("output.jpg", image);
	//
	return 0;
}