/*
 * =====================================================================================
 *
 *       filename:  geekeyedll.cpp
 *
 *    description:  caffe interface
 *
 *        version:  1.0
 *        created:  2016-01-23
 *       revision:  none
 *       compiler:  g++
 *
 *         author:  tangyuan
 *        company:  in66.com
 *
 *      copyright:  2016 itugo Inc. All Rights Reserved.
 *
 * =====================================================================================
 */

#include "geekeyedll/geekeye_dll.h"

//#define ANDROID_LOG
#ifdef ANDROID_LOG
#include <android/log.h>
#endif

namespace geekeyelab {
    
    //
    static bool predict_comp(const std::pair <int, float> elem1, const std::pair <int, float> elem2)
    {
        return (elem1.second > elem2.second);
    }
    
    
    
    
    
    //
    int GeekeyeDLL::init(const std::string& model_file,
                         const int          use_gpu,
                         const int          device_id)
    {
        
        // set gpu device
        _use_gpu = 0;
        _device_id = 0;
        
        
        // data copy
        //std::cout<<"reset."<<std::endl;
        //_net_dl.reset(new caffe::Net<float>(model_file, 1));
        //caffe::shared_ptr<caffe::Net<float> > _net_tmp(new caffe::Net<float>(model_file));
        _net_dl.reset(new caffe::Net<float>());
        //_net_dl->CopyFrom(model_file);
		_net_dl->CopyFrom_int8(model_file);
		        
        // get net param
        std::vector<caffe::Blob<float>*> input_blobs = _net_dl->input_blobs();
        //std::cout<<"input_blobs.size():"<<input_blobs.size()<<std::endl;
        if(input_blobs.size()>0){
            //std::cout<<"input blob:"<<std::cout<<input_blobs[0]->width()<<"x"<<input_blobs[0]->height()
            //     <<"x"<<input_blobs[0]->channels()<<"x"<<input_blobs[0]->num()<<std::endl;
            _input_blob_width = input_blobs[0]->width();
            _input_blob_height = input_blobs[0]->height();
            _input_blob_channel = input_blobs[0]->channels();
            _input_blob_num = input_blobs[0]->num();
        }
        
        //
        //check_layers();
        //check_blob_names();
        
        //
        _mean_model = 0;
        _init_flag = true;
        _mean_value.assign(_input_blob_channel, 0.0f);
        return 0;
    }
    
    
    
    //
    int GeekeyeDLL::predict(const unsigned char* bgr, int width, int height, std::vector< std::pair<int, float> >& results)
    {
        if(NULL == bgr) return -1;
        if(width != _input_blob_width || height !=  _input_blob_height) return -2;
        
        //
        caffe::Blob<float> image_blob( 1, _input_blob_channel, _input_blob_height, _input_blob_width );
        float pixel;
        float* blob_ptr = image_blob.mutable_cpu_data();
        
        {
            for (int c = 0; c < _input_blob_channel; ++c) {
                for (int h = 0; h < _input_blob_height; ++h) {
                    for (int w = 0, p = 0; w < _input_blob_width; ++w, p+=_input_blob_channel) {
                        
                        pixel = (float)bgr[h*_input_blob_width*_input_blob_channel + p + c];
                        *blob_ptr = (pixel - _mean_value[c]);
                        blob_ptr++;
                    }
                }
            }
        }
        
        // input layer
        std::vector<caffe::Blob<float>*> input_blobs = _net_dl->input_blobs();
        // image blob to input layer
        for (int i = 0; i < input_blobs.size(); ++i) {
            caffe::caffe_copy(input_blobs[i]->count(), image_blob.mutable_cpu_data(),input_blobs[i]->mutable_cpu_data());
        }
        
        
        // do forward
        float iter_loss = 0.0;
        std::vector<caffe::Blob<float>*> output_blobs = _net_dl->Forward(input_blobs, &iter_loss);
        if(output_blobs.size() < 1){
            std::cout<<"do forward failed!"<<std::endl;
            exit(0);
        }
        _fc_output_num = output_blobs[0]->count();
        
        // output prob
        //if(1 == output_blobs.size())
        {
            for (int k=0; k < _fc_output_num; ++k )
            {
                results.push_back( std::make_pair( k, output_blobs[0]->cpu_data()[k] ) );
            }
        }
        
        //sort label result
        sort(results.begin(), results.end(), predict_comp);
        std::cout<<" "<<results[0].first<<" - "<<results[0].second<<std::endl;
        
        return 0;
    }
    
#ifdef USE_OPENCV
    int GeekeyeDLL::predict(const std::string& image_file, std::vector< std::pair<int, float> >& results)
    {
        //
        cv::Mat src = cv::imread(image_file, 1);
        if(!src.data){
            std::cout<<"cannot load image:"<<image_file<<std::endl;
            return -1;
        }
        //
        cv::Mat im;
        cv::resize(src, im, cv::Size(_input_blob_width, _input_blob_height));
        if (im.channels() == 4)
            cv::cvtColor(im, im, cv::COLOR_BGRA2BGR);
        //
        predict(im.data, im.cols, im.rows, results);


        return 0;
    }
#endif 
    
    //
    void GeekeyeDLL::release()
    {
        if (_net_dl) {
            _net_dl.reset();
            _init_flag = false;
        }
        
    }
    
    
    //
    void GeekeyeDLL::set_gpu(int c){
        _use_gpu = c;
    }
    
    //
    void GeekeyeDLL::set_device(int c){
        _device_id = c;
    }
    
    
    //
    int GeekeyeDLL::set_mean_value(const std::vector<float>& mean_value){
        if(_input_blob_channel != mean_value.size()){
            return -1;
        }
        for(int i = 0; i<mean_value.size(); i++)
            _mean_value[i] = mean_value[i];
        _mean_model = 2;
        return 0;
    }
    
    //
    void GeekeyeDLL::check_layers()
    {
        //
        const std::vector<caffe::shared_ptr<caffe::Layer<float> > >& layers = _net_dl->layers();
        const std::vector<std::string>& layer_names = _net_dl->layer_names();
        const std::vector<std::string>& blob_names = _net_dl->blob_names();
        int num_layers = 0;
        {
            std::string prev_layer_name = "";
            for (unsigned int i = 0; i < layers.size(); ++i) {
                std::vector<caffe::shared_ptr<caffe::Blob<float> > >& layer_blobs = layers[i]->blobs();
                if (layer_blobs.size() == 0) {
                    std::cout<<"layer_blobs.size() == 0, continue;"<<std::endl;
                    continue;
                }
                std::cout<<"layer index:"<<i<<std::endl;
                std::cout<<"layer name:"<<layer_names[i]<<std::endl;
                std::cout<<"layer_blobs.size():"<<layer_blobs.size()<<std::endl;
                for(int b = 0; b < layer_blobs.size(); b++)
                    std::cout<<layer_blobs[b]->width()<<"x"<<layer_blobs[b]->height()
                    <<"x"<<layer_blobs[b]->channels()<<"x"<<layer_blobs[b]->num()<<std::endl;
                //
                caffe::shared_ptr<caffe::Blob<float> > feature_blob = layer_blobs[0];
                std::cout<<"param:"<<feature_blob->num()<<" "<<feature_blob->count()<<std::endl;
            }//for-i
        }
        
    }
    
    //
    void GeekeyeDLL::check_blob_names(){
        const std::vector<std::string>& blob_names = _net_dl->blob_names();
        std::cout<<"blob names size:"<<blob_names.size()<<std::endl;
        for(int i = 0; i<blob_names.size(); i++){
            caffe::shared_ptr<caffe::Blob<float> > feature_blob;
            feature_blob = _net_dl->blob_by_name(blob_names[i]);
            std::cout<<"("<<i<<","<<blob_names[i]<<":"<<feature_blob->width()<<"x"<<feature_blob->height()
            <<"x"<<feature_blob->channels()<<"x"<<feature_blob->num()<<") ";
        }
        std::cout<<std::endl;
    }
    
    
    
    
    int GeekeyeDLL::get_layer_params(const std::string& layer_name, std::vector< std::vector<float> >& params_all)
    {
        //
        const std::vector<caffe::shared_ptr<caffe::Layer<float> > >& layers = _net_dl->layers();
        const std::vector<std::string>& layer_names = _net_dl->layer_names();
        const std::vector<std::string>& blob_names = _net_dl->blob_names();
        int num_layers = 0;
        {
            std::string prev_layer_name = "";
            for (unsigned int i = 0; i < layers.size(); ++i) {
                if(layer_names[i] != layer_name) continue;
                std::vector<caffe::shared_ptr<caffe::Blob<float> > >& layer_blobs = layers[i]->blobs();
                //
                caffe::shared_ptr<caffe::Blob<float> > feature_blob = layer_blobs[0];
                std::cout<<"param:"<<feature_blob->num()<<" "<<feature_blob->count()<<std::endl;
                int batch_size = feature_blob->num();
                int dim_features = feature_blob->count() / batch_size;
                std::vector<float> params_one;
                const float* feature_blob_data;
                params_all.clear();
                for (int i = 0; i < batch_size; ++i) {	  
                    feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(i);
                    std::copy ( feature_blob_data, feature_blob_data + dim_features, std::back_inserter(params_one));
                    params_all.push_back( params_one );
                    params_one.clear();
                }
                
                
            }//for-i
        }
        
        return 0;
    }
    
}//namespace
