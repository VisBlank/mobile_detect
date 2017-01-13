#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

//#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
//#include "caffe/parallel.hpp"
//#include "caffe/util/hdf5.hpp"
//#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/util/upgrade_proto.hpp"

//#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/fast_rcnn_layers.hpp" 


//#include <android/log.h>
//#define TAG    "log-jni" // 这个是自定义的LOG的标识
//#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__) // 定义LOGD类型


//#include "caffe/proto/caffe.pb.h"

namespace caffe {
    

    
    template <typename Dtype>
    Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
        //CHECK_GE(start, 0);
        //CHECK_LT(end, layers_.size());
        Dtype loss = 0;
        //if (debug_info_)
        {
            //for (int i = 0; i < net_input_blobs_.size(); ++i) {
            //  InputDebugInfo(i);
            //}
        }
        for (int i = start; i <= end; ++i) {
            // LOG(ERROR) << "Forwarding " << layer_names_[i];
            //std::cout<<i<<"-th forward:"<<layers_[i]->type()<<" - "<<layer_names_[i].c_str()<<std::endl;
            Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
            loss += layer_loss;
            //if (debug_info_)
            // ForwardDebugInfo(i);
            
        }
        
        //std::cout<<"ForwardFromTo end."<<std::endl;
        return loss;
    }
    
    template <typename Dtype>
    Dtype Net<Dtype>::ForwardFrom(int start) {
        return ForwardFromTo(start, layers_.size() - 1);
    }
    
    template <typename Dtype>
    Dtype Net<Dtype>::ForwardTo(int end) {
        return ForwardFromTo(0, end);
    }
    
    template <typename Dtype>
    const vector<Blob<Dtype>*>& Net<Dtype>::ForwardPrefilled(Dtype* loss) {
        
        if (loss != NULL) {
            *loss = ForwardFromTo(0, layers_.size() - 1);
        } else {
            ForwardFromTo(0, layers_.size() - 1);
        }
        
        return net_output_blobs_;
    }
    
    template <typename Dtype>
    const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
                                                    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
        // Copy bottom to internal bottom
        for (int i = 0; i < bottom.size(); ++i) {
            net_input_blobs_[i]->CopyFrom(*bottom[i]);
        }
        return ForwardPrefilled(loss);
    }
    

    
    template <typename Dtype>
    bool Net<Dtype>::has_blob(const string& blob_name) const {
        return blob_names_index_.find(blob_name) != blob_names_index_.end();
    }
    
    template <typename Dtype>
    const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
                                                            const string& blob_name) const {
        shared_ptr<Blob<Dtype> > blob_ptr;
        if (has_blob(blob_name)) {
            blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
        } else {
            blob_ptr.reset((Blob<Dtype>*)(NULL));
            //LOG(WARNING) << "Unknown blob name " << blob_name;
        }
        return blob_ptr;
    }
    
    template <typename Dtype>
    bool Net<Dtype>::has_layer(const string& layer_name) const {
        return layer_names_index_.find(layer_name) != layer_names_index_.end();
    }
    
    template <typename Dtype>
    const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
                                                              const string& layer_name) const {
        shared_ptr<Layer<Dtype> > layer_ptr;
        if (has_layer(layer_name)) {
            layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
        } else {
            layer_ptr.reset((Layer<Dtype>*)(NULL));
            //LOG(WARNING) << "Unknown layer name " << layer_name;
        }
        return layer_ptr;
    }
    
    //
    template <typename Dtype>
    void Net<Dtype>::CopyFrom(const std::string& m_file)
    {
        std::cout<<"copy net."<<std::endl;
        phase_ = caffe::TEST_;
        
        std::ifstream is(m_file.c_str(), std::ios::binary);
        if(!is.is_open()){
            std::cout<<"cannot load:"<<m_file<<std::endl;
            return ;
        }
        
        uint64_t blobs_size;
        int n, c, w, h;
        blobs_size = read_real<uint64_t>(is);
        this->blob_names_.resize(blobs_size);
        for(int b = 0; b<blobs_size; b++){
            shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
            this->blobs_.push_back(blob_pointer);
            //
            read_string(is, this->blob_names_[b]);
            n=read_real<int>(is);
            c=read_real<int>(is);
            w=read_real<int>(is);
            h=read_real<int>(is);
            blob_pointer->Reshape(n, c, w, h);
        }
        
        //
        net_input_blobs_.push_back(blobs_[0].get());
        net_output_blobs_.push_back(blobs_[blobs_.size()-1].get());
        
        //
        uint64_t layers_size;
        uint64_t sz;
        layers_size = read_real<uint64_t>(is);
        this->layer_names_.resize(layers_size);
        this->bottom_id_vecs_.resize(layers_size);
        this->top_id_vecs_.resize(layers_size);
        this->bottom_vecs_.resize(layers_size);
        this->top_vecs_.resize(layers_size);
        
        for (unsigned int i = 0; i < layers_size; ++i) {
            //shared_ptr<Layer<Dtype> > layer = other->layers()[i];
            //
            std::string layer_type;
            read_string(is, layer_type);
            read_string(is, this->layer_names_[i]);
            
            
            sz = read_real<uint64_t>(is);
            this->bottom_id_vecs_[i].resize(sz);
            for(int b = 0; b<sz; b++){
                this->bottom_id_vecs_[i][b] = read_real<int>(is);
                this->bottom_vecs_[i].push_back( this->blobs_[this->bottom_id_vecs_[i][b]].get() );
            }
            
            sz = read_real<uint64_t>(is);
            this->top_id_vecs_[i].resize(sz);
            for(int b = 0; b<sz; b++){
                this->top_id_vecs_[i][b] = read_real<int>(is);
                this->top_vecs_[i].push_back( this->blobs_[this->top_id_vecs_[i][b]].get() );
            }
            
            if( layer_type == "Split"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                shared_ptr<SplitLayer<Dtype> > sl(new SplitLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                layers_.push_back(sl);
            }
            else if( layer_type == "ReLU"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                Dtype p = read_real<Dtype>(is);
                shared_ptr<ReLULayer<Dtype> > sl(new ReLULayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);
            }
            else if( layer_type == "Concat"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                ConcatParam p;
                p.concat_axis = read_real<int>(is);
                shared_ptr<ConcatLayer<Dtype> > sl(new ConcatLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);
            }
            else if( layer_type == "Dropout"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                shared_ptr<DropoutLayer<Dtype> > sl(new DropoutLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                layers_.push_back(sl);
            }
            else if( layer_type == "Pooling"){
                //std::cout<<"cp:"<<layer->type()<<":"<<layer->blobs().size()<<std::endl;

                PoolingParam pt;
                pt.global_pooling = (bool)read_real<int>(is);
                pt.pool_method  = (bool)read_real<int>(is);
                pt.kernel_h = read_real<int>(is);
                pt.kernel_w = read_real<int>(is);
                pt.pad_h = read_real<int>(is);
                pt.pad_w = read_real<int>(is);
                pt.stride_h = read_real<int>(is);
                pt.stride_w = read_real<int>(is);
                //
                shared_ptr<PoolingLayer<Dtype> > sl(new PoolingLayer<Dtype>( ) );
                //sl->CopyFrom(layer->layer_param());
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(pt);
                layers_.push_back(sl);
            }
            else if( layer_type == "Softmax"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                int p = read_real<int>(is);
                shared_ptr<SoftmaxLayer<Dtype> > sl(new SoftmaxLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);
            }
            else if( layer_type == "Convolution"){
                //std::cout<<i<<" cp:"<<layer->type()<<std::endl;

                // --
                ConvolutionParam pt;
                pt.force_nd_im2col = (bool)read_real<int>(is);
                pt.channel_axis = read_real<int>(is);
                pt.num_spatial_axes = read_real<int>(is);
                //
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.kernel_shape.Reshape(n, c, w, h);
                read_real<int>(is, pt.kernel_shape.mutable_cpu_data(), pt.kernel_shape.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.stride.Reshape(n, c, w, h);
                read_real<int>(is, pt.stride.mutable_cpu_data(), pt.stride.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.pad.Reshape(n, c, w, h);
                read_real<int>(is, pt.pad.mutable_cpu_data(), pt.pad.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.dilation.Reshape(n, c, w, h);
                read_real<int>(is, pt.dilation.mutable_cpu_data(), pt.dilation.count() );
                //
                pt.is_1x1 = (bool)read_real<int>(is);
                pt.channels = read_real<int>(is);
                pt.num_output = read_real<int>(is);
                pt.group = read_real<int>(is);
                pt.conv_out_channels = read_real<int>(is);
                pt.conv_in_channels = read_real<int>(is);
                pt.bias_term = (bool)read_real<int>(is);
                pt.kernel_dim = read_real<int>(is);
                //
                sz = read_real<uint64_t>(is);
                vector<shared_ptr<Blob<Dtype> > > layer_blobs;
                for(int b = 0; b<sz; b++){
                    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
                    layer_blobs.push_back(blob_pointer);
                    //
                    n=read_real<int>(is);
                    c=read_real<int>(is);
                    w=read_real<int>(is);
                    h=read_real<int>(is);
                    blob_pointer->Reshape(n, c, w, h);
                    read_real<Dtype>(is, blob_pointer->mutable_cpu_data(), blob_pointer->count() );
                }
                //
                shared_ptr<ConvolutionLayer<Dtype> > cl(new ConvolutionLayer<Dtype>(  ) );
                cl->CopyFrom(pt);
                cl->CopyBlob(layer_blobs);
                //
                layers_.push_back(cl);
            }
            
        }
        is.close();
        
        
    }

	//
    template <typename Dtype>
    void Net<Dtype>::CopyFrom_int8(const std::string& m_file)
    {
        std::cout<<"copy net."<<std::endl;
        phase_ = caffe::TEST_;
        
        std::ifstream is(m_file.c_str(), std::ios::binary);
        if(!is.is_open()){
            std::cout<<"cannot load:"<<m_file<<std::endl;
            return ;
        }
        
        uint64_t blobs_size;
        int n, c, w, h;
        blobs_size = read_real<uint64_t>(is);
        this->blob_names_.resize(blobs_size);
        for(int b = 0; b<blobs_size; b++){
            shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
            this->blobs_.push_back(blob_pointer);
            //
            read_string(is, this->blob_names_[b]);
            n=read_real<int>(is);
            c=read_real<int>(is);
            w=read_real<int>(is);
            h=read_real<int>(is);
            blob_pointer->Reshape(n, c, w, h);
        }
        
        //
        net_input_blobs_.push_back(blobs_[0].get());
        net_output_blobs_.push_back(blobs_[blobs_.size()-1].get());
        
        //
        uint64_t layers_size;
        uint64_t sz;
        layers_size = read_real<uint64_t>(is);
        this->layer_names_.resize(layers_size);
        this->bottom_id_vecs_.resize(layers_size);
        this->top_id_vecs_.resize(layers_size);
        this->bottom_vecs_.resize(layers_size);
        this->top_vecs_.resize(layers_size);
        
        for (unsigned int i = 0; i < layers_size; ++i) {
            //shared_ptr<Layer<Dtype> > layer = other->layers()[i];
            //
            std::string layer_type;
            read_string(is, layer_type);
            read_string(is, this->layer_names_[i]);
            
            
            sz = read_real<uint64_t>(is);
            this->bottom_id_vecs_[i].resize(sz);
            for(int b = 0; b<sz; b++){
                this->bottom_id_vecs_[i][b] = read_real<int>(is);
                this->bottom_vecs_[i].push_back( this->blobs_[this->bottom_id_vecs_[i][b]].get() );
            }
            
            sz = read_real<uint64_t>(is);
            this->top_id_vecs_[i].resize(sz);
            for(int b = 0; b<sz; b++){
                this->top_id_vecs_[i][b] = read_real<int>(is);
                this->top_vecs_[i].push_back( this->blobs_[this->top_id_vecs_[i][b]].get() );
            }
            
            if( layer_type == "Split"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                shared_ptr<SplitLayer<Dtype> > sl(new SplitLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                layers_.push_back(sl);
            }
            else if( layer_type == "ReLU"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                Dtype p = read_real<Dtype>(is);
                shared_ptr<ReLULayer<Dtype> > sl(new ReLULayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);
            }
            else if( layer_type == "Concat"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                ConcatParam p;
                p.concat_axis = read_real<int>(is);
                shared_ptr<ConcatLayer<Dtype> > sl(new ConcatLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);
            }
            else if( layer_type == "Dropout"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                shared_ptr<DropoutLayer<Dtype> > sl(new DropoutLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                layers_.push_back(sl);
            }
            else if( layer_type == "Pooling"){
                //std::cout<<"cp:"<<layer->type()<<":"<<layer->blobs().size()<<std::endl;

                PoolingParam pt;
                pt.global_pooling = (bool)read_real<int>(is);
                pt.pool_method  = (bool)read_real<int>(is);
                pt.kernel_h = read_real<int>(is);
                pt.kernel_w = read_real<int>(is);
                pt.pad_h = read_real<int>(is);
                pt.pad_w = read_real<int>(is);
                pt.stride_h = read_real<int>(is);
                pt.stride_w = read_real<int>(is);
                //
                shared_ptr<PoolingLayer<Dtype> > sl(new PoolingLayer<Dtype>( ) );
                //sl->CopyFrom(layer->layer_param());
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(pt);
                layers_.push_back(sl);
            }
            else if( layer_type == "Softmax"){
                //std::cout<<"cp:"<<layer->type()<<std::endl;
                int p = read_real<int>(is);
                shared_ptr<SoftmaxLayer<Dtype> > sl(new SoftmaxLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);
            }
            else if( layer_type == "Convolution"){
                //std::cout<<i<<" cp:"<<layer->type()<<std::endl;

                // --
                ConvolutionParam pt;
                pt.force_nd_im2col = (bool)read_real<int>(is);
                pt.channel_axis = read_real<int>(is);
                pt.num_spatial_axes = read_real<int>(is);
                //
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.kernel_shape.Reshape(n, c, w, h);
                read_real<int>(is, pt.kernel_shape.mutable_cpu_data(), pt.kernel_shape.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.stride.Reshape(n, c, w, h);
                read_real<int>(is, pt.stride.mutable_cpu_data(), pt.stride.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.pad.Reshape(n, c, w, h);
                read_real<int>(is, pt.pad.mutable_cpu_data(), pt.pad.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.dilation.Reshape(n, c, w, h);
                read_real<int>(is, pt.dilation.mutable_cpu_data(), pt.dilation.count() );
                //
                pt.is_1x1 = (bool)read_real<int>(is);
                pt.channels = read_real<int>(is);
                pt.num_output = read_real<int>(is);
                pt.group = read_real<int>(is);
                pt.conv_out_channels = read_real<int>(is);
                pt.conv_in_channels = read_real<int>(is);
                pt.bias_term = (bool)read_real<int>(is);
                pt.kernel_dim = read_real<int>(is);
                //
                //
                //sz = read_real<uint64_t>(is);
                //vector<shared_ptr<Blob<Dtype> > > layer_blobs;
                //for(int b = 0; b<sz; b++){
                //    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
                //    layer_blobs.push_back(blob_pointer);
                    //
                //    n=read_real<int>(is);
                //    c=read_real<int>(is);
                //    w=read_real<int>(is);
                //    h=read_real<int>(is);
                //    blob_pointer->Reshape(n, c, w, h);
                //    read_real<Dtype>(is, blob_pointer->mutable_cpu_data(), blob_pointer->count() );
                //}

				//read data and change
				vector<shared_ptr<Blob<Dtype> > > layer_blobs;
				blob_uchar2float(is, layer_blobs);
				
                //
                shared_ptr<ConvolutionLayer<Dtype> > cl(new ConvolutionLayer<Dtype>(  ) );
                cl->CopyFrom(pt);
                cl->CopyBlob(layer_blobs);
                //
                layers_.push_back(cl);
            }
            
        }
        is.close();
        
        
    }

	//add by chigo for detect
    template <typename Dtype>
    void Net<Dtype>::CopyFrom_detect(const std::string& m_file)
    {
        std::cout<<"copy net."<<std::endl;
        phase_ = caffe::TEST_;
        
        std::ifstream is(m_file.c_str(), std::ios::binary);
        if(!is.is_open()){
            std::cout<<"cannot load:"<<m_file<<std::endl;
            return ;
        }
        
        uint64_t blobs_size;
        int n, c, w, h;
        blobs_size = read_real<uint64_t>(is);
        this->blob_names_.resize(blobs_size);
        for(int b = 0; b<blobs_size; b++){
            shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
            this->blobs_.push_back(blob_pointer);
            //
            read_string(is, this->blob_names_[b]);
            n=read_real<int>(is);
            c=read_real<int>(is);
            w=read_real<int>(is);
            h=read_real<int>(is);
            blob_pointer->Reshape(n, c, w, h);
        }
        
        //
        //printf( "get blobs data...\n" );
		net_input_blobs_.push_back(blobs_[0].get());	//input: "data"
		net_input_blobs_.push_back(blobs_[1].get());	//input: "im_info"
		net_output_blobs_.push_back(blobs_[blobs_.size()-1].get());		//top: "cls_prob",Softmax
		net_output_blobs_.push_back(blobs_[blobs_.size()-2].get());		//top: "bbox_pred",
		net_output_blobs_.push_back(blobs_[blobs_.size()-12].get());	//top: "rois",ProposalLayer;"Split Layer"
		//printf( "get blobs data end!!\n" );
        
        //
        uint64_t layers_size;
        uint64_t sz;
        layers_size = read_real<uint64_t>(is);
        this->layer_names_.resize(layers_size);
        this->bottom_id_vecs_.resize(layers_size);
        this->top_id_vecs_.resize(layers_size);
        this->bottom_vecs_.resize(layers_size);
        this->top_vecs_.resize(layers_size);

		
        
        for (unsigned int i = 0; i < layers_size; ++i) {
            //shared_ptr<Layer<Dtype> > layer = other->layers()[i];
            //
            std::string layer_type;
            read_string(is, layer_type);
            read_string(is, this->layer_names_[i]);
            
            
            sz = read_real<uint64_t>(is);
            this->bottom_id_vecs_[i].resize(sz);
            for(int b = 0; b<sz; b++){
                this->bottom_id_vecs_[i][b] = read_real<int>(is);
                this->bottom_vecs_[i].push_back( this->blobs_[this->bottom_id_vecs_[i][b]].get() );
            }
            
            sz = read_real<uint64_t>(is);
            this->top_id_vecs_[i].resize(sz);
            for(int b = 0; b<sz; b++){
                this->top_id_vecs_[i][b] = read_real<int>(is);
                this->top_vecs_[i].push_back( this->blobs_[this->top_id_vecs_[i][b]].get() );
            }
            
            if( layer_type == "Split"){
                printf( "i:%d,type:%s\n", i, layer_type.c_str() );
                shared_ptr<SplitLayer<Dtype> > sl(new SplitLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                layers_.push_back(sl);
            }
            else if( layer_type == "ReLU"){
                printf( "i:%d,type:%s\n", i, layer_type.c_str() );
                Dtype p = read_real<Dtype>(is);
                shared_ptr<ReLULayer<Dtype> > sl(new ReLULayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);
            }
            else if( layer_type == "Concat"){
                printf( "i:%d,type:%s\n", i, layer_type.c_str() );
                ConcatParam p;
                p.concat_axis = read_real<int>(is);
                shared_ptr<ConcatLayer<Dtype> > sl(new ConcatLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);
            }
            else if( layer_type == "Dropout"){
                printf( "i:%d,type:%s\n", i, layer_type.c_str() );
                shared_ptr<DropoutLayer<Dtype> > sl(new DropoutLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                layers_.push_back(sl);
            }
            else if( layer_type == "Pooling"){
                printf( "i:%d,type:%s\n", i, layer_type.c_str() );

                PoolingParam pt;
                pt.global_pooling = (bool)read_real<int>(is);
                pt.pool_method  = (bool)read_real<int>(is);
                pt.kernel_h = read_real<int>(is);
                pt.kernel_w = read_real<int>(is);
                pt.pad_h = read_real<int>(is);
                pt.pad_w = read_real<int>(is);
                pt.stride_h = read_real<int>(is);
                pt.stride_w = read_real<int>(is);
                //
                shared_ptr<PoolingLayer<Dtype> > sl(new PoolingLayer<Dtype>( ) );
                //sl->CopyFrom(layer->layer_param());
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(pt);
                layers_.push_back(sl);
            }
            else if( layer_type == "Softmax"){
                printf( "i:%d,type:%s\n", i, layer_type.c_str() );
                int p = read_real<int>(is);
                shared_ptr<SoftmaxLayer<Dtype> > sl(new SoftmaxLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);
            }
            else if( layer_type == "Convolution"){
                printf( "i:%d,type:%s\n", i, layer_type.c_str() );

                // --
                ConvolutionParam pt;
                pt.force_nd_im2col = (bool)read_real<int>(is);
                pt.channel_axis = read_real<int>(is);
                pt.num_spatial_axes = read_real<int>(is);
                //
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.kernel_shape.Reshape(n, c, w, h);
                read_real<int>(is, pt.kernel_shape.mutable_cpu_data(), pt.kernel_shape.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.stride.Reshape(n, c, w, h);
                read_real<int>(is, pt.stride.mutable_cpu_data(), pt.stride.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.pad.Reshape(n, c, w, h);
                read_real<int>(is, pt.pad.mutable_cpu_data(), pt.pad.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.dilation.Reshape(n, c, w, h);
                read_real<int>(is, pt.dilation.mutable_cpu_data(), pt.dilation.count() );
                //
                pt.is_1x1 = (bool)read_real<int>(is);
                pt.channels = read_real<int>(is);
                pt.num_output = read_real<int>(is);
                pt.group = read_real<int>(is);
                pt.conv_out_channels = read_real<int>(is);
                pt.conv_in_channels = read_real<int>(is);
                pt.bias_term = (bool)read_real<int>(is);
                pt.kernel_dim = read_real<int>(is);
                //
                sz = read_real<uint64_t>(is);
                vector<shared_ptr<Blob<Dtype> > > layer_blobs;
                for(int b = 0; b<sz; b++){
                    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
                    layer_blobs.push_back(blob_pointer);
                    //
                    n=read_real<int>(is);
                    c=read_real<int>(is);
                    w=read_real<int>(is);
                    h=read_real<int>(is);
                    blob_pointer->Reshape(n, c, w, h);
                    read_real<Dtype>(is, blob_pointer->mutable_cpu_data(), blob_pointer->count() );
                }
                //
                shared_ptr<ConvolutionLayer<Dtype> > cl(new ConvolutionLayer<Dtype>(  ) );
                cl->CopyFrom(pt);
                cl->CopyBlob(layer_blobs);
                //
                layers_.push_back(cl);
            }
			else if( layer_type == "Reshape"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );

				int num_shape;
				vector<int> shape_;

				ReshapeParam pt;
                pt.axis_ = read_real<int>(is);
                pt.num_axes_ = read_real<int>(is);
				num_shape = read_real<uint64_t>(is);
				
				printf( "type:%s,axis:%d,num_axes:%d,shape.size:%d,", 
					layer_type.c_str(),pt.axis_,pt.num_axes_,num_shape );
				for(int b = 0; b < num_shape; b++){
					shape_.push_back(read_real<int>(is));
					printf( "%d_", shape_[b] );
            	}
				printf( "\n");
                pt.shape_ = shape_;
				
                //
                shared_ptr<ReshapeLayer<Dtype> > sl(new ReshapeLayer<Dtype>( ) );
                sl->CopyFrom(pt);
                sl->SetUp(this->bottom_vecs()[i], this->top_vecs()[i]);
                
                layers_.push_back(sl);
	        }
	        else if( layer_type == "Deconvolution"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );

				// --
                ConvolutionParam pt;
                pt.force_nd_im2col = (bool)read_real<int>(is);
                pt.channel_axis = read_real<int>(is);
                pt.num_spatial_axes = read_real<int>(is);
                //
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.kernel_shape.Reshape(n, c, w, h);
                read_real<int>(is, pt.kernel_shape.mutable_cpu_data(), pt.kernel_shape.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.stride.Reshape(n, c, w, h);
                read_real<int>(is, pt.stride.mutable_cpu_data(), pt.stride.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.pad.Reshape(n, c, w, h);
                read_real<int>(is, pt.pad.mutable_cpu_data(), pt.pad.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.dilation.Reshape(n, c, w, h);
                read_real<int>(is, pt.dilation.mutable_cpu_data(), pt.dilation.count() );
                //
                pt.is_1x1 = (bool)read_real<int>(is);
                pt.channels = read_real<int>(is);
                pt.num_output = read_real<int>(is);
                pt.group = read_real<int>(is);
                pt.conv_out_channels = read_real<int>(is);
                pt.conv_in_channels = read_real<int>(is);
                pt.bias_term = (bool)read_real<int>(is);
                pt.kernel_dim = read_real<int>(is);
                //
                sz = read_real<uint64_t>(is);
                vector<shared_ptr<Blob<Dtype> > > layer_blobs;
                for(int b = 0; b<sz; b++){
                    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
                    layer_blobs.push_back(blob_pointer);
                    //
                    n=read_real<int>(is);
                    c=read_real<int>(is);
                    w=read_real<int>(is);
                    h=read_real<int>(is);
                    blob_pointer->Reshape(n, c, w, h);
                    read_real<Dtype>(is, blob_pointer->mutable_cpu_data(), blob_pointer->count() );
                }
                //
                shared_ptr<DeconvolutionLayer<Dtype> > cl(new DeconvolutionLayer<Dtype>(  ) );
                cl->CopyFrom(pt);
                cl->CopyBlob(layer_blobs);
                //
                layers_.push_back(cl);
	        }
			else if( layer_type == "ProposalLayer"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );

				int b,num_ratios,num_scales;
				vector<Dtype> vecRatios;
				vector<Dtype> vecScales;				

				ProposalParam pt;
                pt.base_size_ = read_real<int>(is);
                pt.feat_stride_ = read_real<int>(is);
				pt.pre_nms_topn_ = read_real<int>(is);
                pt.post_nms_topn_ = read_real<int>(is);
				pt.nms_thresh_ = read_real<Dtype>(is);
                pt.min_size_ = read_real<int>(is);
				num_ratios = read_real<uint64_t>(is);

				printf( "type:%s,base_size:%d,feat_stride:%d,pre_nms_topn:%d,post_nms_topn:%d,nms_thresh:%.4f,min_size:%d,vecRatios.size:%d,", 
					layer_type.c_str(),pt.base_size_,pt.feat_stride_,pt.pre_nms_topn_,pt.post_nms_topn_,
					pt.nms_thresh_,pt.min_size_,num_ratios );
				
				for(b = 0; b < num_ratios; b++){
					vecRatios.push_back(read_real<Dtype>(is));
					printf( "%.4f_", vecRatios[b] );
            	}
				printf( "\n");
                pt.ratios_ = vecRatios;

				num_scales = read_real<uint64_t>(is);

				printf( ",vecScales.size:%d,",num_scales);
				for(b = 0; b < num_scales; b++){
					vecScales.push_back(read_real<Dtype>(is));
					printf( "%.4f_", vecScales[b] );
            	}
				printf( "\n");
                pt.scales_ = vecScales;
				
                //
                shared_ptr<ProposalLayer<Dtype> > sl(new ProposalLayer<Dtype>( ) );
                sl->CopyFrom(pt);
                sl->SetUp(this->bottom_vecs()[i], this->top_vecs()[i]);
                
                layers_.push_back(sl);
	        }
			else if( layer_type == "ROIPooling"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );

				ROIPoolingParam pt;
                pt.pooled_width_ = read_real<int>(is);
                pt.pooled_height_ = read_real<int>(is);
				pt.spatial_scale_ = (float)read_real<Dtype>(is);
				
				printf( "type:%s,pooled_width:%d,pooled_height:%d,spatial_scale:%.4f\n", 
					layer_type.c_str(),pt.pooled_width_,pt.pooled_height_,pt.spatial_scale_ );
				
                //
                shared_ptr<ROIPoolingLayer<Dtype> > sl(new ROIPoolingLayer<Dtype>( ) );
                sl->CopyFrom(pt);
                sl->SetUp(this->bottom_vecs()[i], this->top_vecs()[i]);
                
                layers_.push_back(sl);
	        }
			else if( layer_type == "InnerProduct"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );

				InnerProductParam pt;
                pt.bias_term_ = (bool)read_real<int>(is);
                pt.transpose_ = (bool)read_real<int>(is);
				pt.num_output_ = read_real<int>(is);
                pt.axis_ = read_real<int>(is);

				printf( "type:%s,bias_term:%d,transpose:%d,num_output:%d,axis:%d\n", 
					layer_type.c_str(),(int)pt.bias_term_,(int)pt.transpose_,pt.num_output_,pt.axis_ );

				//
                sz = read_real<uint64_t>(is);
                vector<shared_ptr<Blob<Dtype> > > layer_blobs;
                for(int b = 0; b<sz; b++){
                    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
                    layer_blobs.push_back(blob_pointer);
                    //
                    n=read_real<int>(is);
                    c=read_real<int>(is);
                    w=read_real<int>(is);
                    h=read_real<int>(is);
                    blob_pointer->Reshape(n, c, w, h);
                    read_real<Dtype>(is, blob_pointer->mutable_cpu_data(), blob_pointer->count() );
                }
				
                //
                shared_ptr<InnerProductLayer<Dtype> > sl(new InnerProductLayer<Dtype>( ) );
                sl->CopyFrom(pt);
				sl->CopyBlob(layer_blobs);
                sl->SetUp(this->bottom_vecs()[i], this->top_vecs()[i]);
                
                layers_.push_back(sl);
	        }
           
        }
        is.close();
        
        
    }

		//add by chigo for detect
    template <typename Dtype>
    void Net<Dtype>::CopyFrom_detect_int8(const std::string& m_file)
    {
        //std::cout<<"copy net."<<std::endl;
		//LOGD("Net::CopyFrom_detect_int8 start...");
        phase_ = caffe::TEST_;

		//LOGD("Net load file:%s",m_file.c_str());
        std::ifstream is(m_file.c_str(), std::ios::binary);
        if(!is.is_open()){
            std::cout<<"cannot load:"<<m_file<<std::endl;
            return ;
        }
		//LOGD("Net load file end!!");

		//size_t-->uint64_t;
        uint64_t blobs_size;
        int n, c, w, h;
        blobs_size = read_real<uint64_t>(is);
		//LOGD("blobs_size:%lld",blobs_size);
		
        this->blob_names_.resize(blobs_size);
        for(int b = 0; b<blobs_size; b++){
            shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
            this->blobs_.push_back(blob_pointer);
            //
            read_string(is, this->blob_names_[b]);
            n=read_real<int>(is);
            c=read_real<int>(is);
            w=read_real<int>(is);
            h=read_real<int>(is);
            blob_pointer->Reshape(n, c, w, h);
			
			//LOGD("b:%d,n:%d,c:%d,w:%d,h:%d,blob_names_:%s", b, n, c, w, h, this->blob_names_[b].c_str());
        }
        
        //
        //printf( "get blobs data...\n" );
        //LOGD("Net get blobs data start...");
		net_input_blobs_.push_back(blobs_[0].get());	//input: "data"
		net_input_blobs_.push_back(blobs_[1].get());	//input: "im_info"
		net_output_blobs_.push_back(blobs_[blobs_.size()-1].get());		//top: "cls_prob",Softmax
		net_output_blobs_.push_back(blobs_[blobs_.size()-2].get());		//top: "bbox_pred",
		net_output_blobs_.push_back(blobs_[blobs_.size()-12].get());	//top: "rois",ProposalLayer;"Split Layer"
		//printf( "get blobs data end!!\n" );
		//LOGD("Net get blobs data end...");
        
        //
        uint64_t layers_size;
        uint64_t sz;
        layers_size = read_real<uint64_t>(is);
        this->layer_names_.resize(layers_size);
        this->bottom_id_vecs_.resize(layers_size);
        this->top_id_vecs_.resize(layers_size);
        this->bottom_vecs_.resize(layers_size);
        this->top_vecs_.resize(layers_size);
        
        for (unsigned int i = 0; i < layers_size; ++i) {
            //shared_ptr<Layer<Dtype> > layer = other->layers()[i];
            //
            std::string layer_type;
            read_string(is, layer_type);
            read_string(is, this->layer_names_[i]);
            
            
            sz = read_real<uint64_t>(is);
            this->bottom_id_vecs_[i].resize(sz);
            for(int b = 0; b<sz; b++){
                this->bottom_id_vecs_[i][b] = read_real<int>(is);
                this->bottom_vecs_[i].push_back( this->blobs_[this->bottom_id_vecs_[i][b]].get() );
            }
            
            sz = read_real<uint64_t>(is);
            this->top_id_vecs_[i].resize(sz);
            for(int b = 0; b<sz; b++){
                this->top_id_vecs_[i][b] = read_real<int>(is);
                this->top_vecs_[i].push_back( this->blobs_[this->top_id_vecs_[i][b]].get() );
            }
            
            if( layer_type == "Split"){
                //printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );
				
                shared_ptr<SplitLayer<Dtype> > sl(new SplitLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
            }
            else if( layer_type == "ReLU"){
                //printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );
				
                Dtype p = read_real<Dtype>(is);
                shared_ptr<ReLULayer<Dtype> > sl(new ReLULayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
            }
            else if( layer_type == "Concat"){
                //printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );
				
                ConcatParam p;
                p.concat_axis = read_real<int>(is);
                shared_ptr<ConcatLayer<Dtype> > sl(new ConcatLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
            }
            else if( layer_type == "Dropout"){
                //printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );
				
                shared_ptr<DropoutLayer<Dtype> > sl(new DropoutLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
            }
            else if( layer_type == "Pooling"){
                //printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );

                PoolingParam pt;
                pt.global_pooling = (bool)read_real<int>(is);
                pt.pool_method  = (bool)read_real<int>(is);
                pt.kernel_h = read_real<int>(is);
                pt.kernel_w = read_real<int>(is);
                pt.pad_h = read_real<int>(is);
                pt.pad_w = read_real<int>(is);
                pt.stride_h = read_real<int>(is);
                pt.stride_w = read_real<int>(is);
                //
                shared_ptr<PoolingLayer<Dtype> > sl(new PoolingLayer<Dtype>( ) );
                //sl->CopyFrom(layer->layer_param());
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(pt);
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
            }
            else if( layer_type == "Softmax"){
                //printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );
				
                int p = read_real<int>(is);
                shared_ptr<SoftmaxLayer<Dtype> > sl(new SoftmaxLayer<Dtype>() );
                //sl->SetUp(other->bottom_vecs()[i], other->top_vecs()[i]);
                sl->CopyFrom(p);
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
            }
            else if( layer_type == "Convolution"){
                //printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );

                // --
                ConvolutionParam pt;
                pt.force_nd_im2col = (bool)read_real<int>(is);
                pt.channel_axis = read_real<int>(is);
                pt.num_spatial_axes = read_real<int>(is);
                //
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.kernel_shape.Reshape(n, c, w, h);
                read_real<int>(is, pt.kernel_shape.mutable_cpu_data(), pt.kernel_shape.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.stride.Reshape(n, c, w, h);
                read_real<int>(is, pt.stride.mutable_cpu_data(), pt.stride.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.pad.Reshape(n, c, w, h);
                read_real<int>(is, pt.pad.mutable_cpu_data(), pt.pad.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.dilation.Reshape(n, c, w, h);
                read_real<int>(is, pt.dilation.mutable_cpu_data(), pt.dilation.count() );
                //
                pt.is_1x1 = (bool)read_real<int>(is);
                pt.channels = read_real<int>(is);
                pt.num_output = read_real<int>(is);
                pt.group = read_real<int>(is);
                pt.conv_out_channels = read_real<int>(is);
                pt.conv_in_channels = read_real<int>(is);
                pt.bias_term = (bool)read_real<int>(is);
                pt.kernel_dim = read_real<int>(is);
                //
                //sz = read_real<uint64_t>(is);
                //vector<shared_ptr<Blob<Dtype> > > layer_blobs;
                //for(int b = 0; b<sz; b++){
                //    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
                //    layer_blobs.push_back(blob_pointer);
                    //
                //    n=read_real<int>(is);
                //    c=read_real<int>(is);
                //    w=read_real<int>(is);
                //    h=read_real<int>(is);
                //    blob_pointer->Reshape(n, c, w, h);
                //    read_real<Dtype>(is, blob_pointer->mutable_cpu_data(), blob_pointer->count() );
                //}

				//read data and change
				vector<shared_ptr<Blob<Dtype> > > layer_blobs;
				blob_uchar2float(is, layer_blobs);
				
                //
                shared_ptr<ConvolutionLayer<Dtype> > cl(new ConvolutionLayer<Dtype>(  ) );
                cl->CopyFrom(pt);
                cl->CopyBlob(layer_blobs);
                //
                layers_.push_back(cl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
            }
			else if( layer_type == "Reshape"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );

				int num_shape;
				vector<int> shape_;

				ReshapeParam pt;
                pt.axis_ = read_real<int>(is);
                pt.num_axes_ = read_real<int>(is);
				num_shape = read_real<uint64_t>(is);
				
				printf( "type:%s,axis:%d,num_axes:%d,shape.size:%d,", 
					layer_type.c_str(),pt.axis_,pt.num_axes_,num_shape );
				for(int b = 0; b < num_shape; b++){
					shape_.push_back(read_real<int>(is));
					printf( "%d_", shape_[b] );
            	}
				printf( "\n");
                pt.shape_ = shape_;
				
                //
                shared_ptr<ReshapeLayer<Dtype> > sl(new ReshapeLayer<Dtype>( ) );
                sl->CopyFrom(pt);
                sl->SetUp(this->bottom_vecs()[i], this->top_vecs()[i]);
                
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
	        }
	        else if( layer_type == "Deconvolution"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );

				// --
                ConvolutionParam pt;
                pt.force_nd_im2col = (bool)read_real<int>(is);
                pt.channel_axis = read_real<int>(is);
                pt.num_spatial_axes = read_real<int>(is);
                //
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.kernel_shape.Reshape(n, c, w, h);
                read_real<int>(is, pt.kernel_shape.mutable_cpu_data(), pt.kernel_shape.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.stride.Reshape(n, c, w, h);
                read_real<int>(is, pt.stride.mutable_cpu_data(), pt.stride.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.pad.Reshape(n, c, w, h);
                read_real<int>(is, pt.pad.mutable_cpu_data(), pt.pad.count() );
                n=read_real<int>(is);
                c=read_real<int>(is);
                w=read_real<int>(is);
                h=read_real<int>(is);
                pt.dilation.Reshape(n, c, w, h);
                read_real<int>(is, pt.dilation.mutable_cpu_data(), pt.dilation.count() );
                //
                pt.is_1x1 = (bool)read_real<int>(is);
                pt.channels = read_real<int>(is);
                pt.num_output = read_real<int>(is);
                pt.group = read_real<int>(is);
                pt.conv_out_channels = read_real<int>(is);
                pt.conv_in_channels = read_real<int>(is);
                pt.bias_term = (bool)read_real<int>(is);
                pt.kernel_dim = read_real<int>(is);
                //
                //
                //sz = read_real<uint64_t>(is);
                //vector<shared_ptr<Blob<Dtype> > > layer_blobs;
                //for(int b = 0; b<sz; b++){
                //    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
                //    layer_blobs.push_back(blob_pointer);
                    //
                //    n=read_real<int>(is);
                //    c=read_real<int>(is);
                //    w=read_real<int>(is);
                //    h=read_real<int>(is);
                //    blob_pointer->Reshape(n, c, w, h);
                //    read_real<Dtype>(is, blob_pointer->mutable_cpu_data(), blob_pointer->count() );
                //}

				//read data and change
				vector<shared_ptr<Blob<Dtype> > > layer_blobs;
				blob_uchar2float(is, layer_blobs);
				
                //
                shared_ptr<DeconvolutionLayer<Dtype> > cl(new DeconvolutionLayer<Dtype>(  ) );
                cl->CopyFrom(pt);
                cl->CopyBlob(layer_blobs);
                //
                layers_.push_back(cl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
	        }
			else if( layer_type == "ProposalLayer"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );

				int b,num_ratios,num_scales;
				vector<Dtype> vecRatios;
				vector<Dtype> vecScales;				

				ProposalParam pt;
                pt.base_size_ = read_real<int>(is);
                pt.feat_stride_ = read_real<int>(is);
				pt.pre_nms_topn_ = read_real<int>(is);
                pt.post_nms_topn_ = read_real<int>(is);
				pt.nms_thresh_ = read_real<Dtype>(is);
                pt.min_size_ = read_real<int>(is);
				num_ratios = read_real<uint64_t>(is);

				printf( "type:%s,base_size:%d,feat_stride:%d,pre_nms_topn:%d,post_nms_topn:%d,nms_thresh:%.4f,min_size:%d,vecRatios.size:%d,", 
					layer_type.c_str(),pt.base_size_,pt.feat_stride_,pt.pre_nms_topn_,pt.post_nms_topn_,
					pt.nms_thresh_,pt.min_size_,num_ratios );
				
				for(b = 0; b < num_ratios; b++){
					vecRatios.push_back(read_real<Dtype>(is));
					printf( "%.4f_", vecRatios[b] );
            	}
				printf( "\n");
                pt.ratios_ = vecRatios;

				num_scales = read_real<uint64_t>(is);

				printf( ",vecScales.size:%d,",num_scales);
				for(b = 0; b < num_scales; b++){
					vecScales.push_back(read_real<Dtype>(is));
					printf( "%.4f_", vecScales[b] );
            	}
				printf( "\n");
                pt.scales_ = vecScales;
				
                //
                shared_ptr<ProposalLayer<Dtype> > sl(new ProposalLayer<Dtype>( ) );
                sl->CopyFrom(pt);
                sl->SetUp(this->bottom_vecs()[i], this->top_vecs()[i]);
                
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
	        }
			else if( layer_type == "ROIPooling"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );

				ROIPoolingParam pt;
                pt.pooled_width_ = read_real<int>(is);
                pt.pooled_height_ = read_real<int>(is);
				pt.spatial_scale_ = (float)read_real<Dtype>(is);
				
				printf( "type:%s,pooled_width:%d,pooled_height:%d,spatial_scale:%.4f\n", 
					layer_type.c_str(),pt.pooled_width_,pt.pooled_height_,pt.spatial_scale_ );
				
                //
                shared_ptr<ROIPoolingLayer<Dtype> > sl(new ROIPoolingLayer<Dtype>( ) );
                sl->CopyFrom(pt);
                sl->SetUp(this->bottom_vecs()[i], this->top_vecs()[i]);
                
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
	        }
			else if( layer_type == "InnerProduct"){
				printf( "i:%d,type:%s\n", i, layer_type.c_str() );
				//LOGD("layer i:%d type:%s start...", i, layer_type.c_str() );

				InnerProductParam pt;
                pt.bias_term_ = (bool)read_real<int>(is);
                pt.transpose_ = (bool)read_real<int>(is);
				pt.num_output_ = read_real<int>(is);
                pt.axis_ = read_real<int>(is);

				printf( "type:%s,bias_term:%d,transpose:%d,num_output:%d,axis:%d\n", 
					layer_type.c_str(),(int)pt.bias_term_,(int)pt.transpose_,pt.num_output_,pt.axis_ );

                //
                //sz = read_real<uint64_t>(is);
                //vector<shared_ptr<Blob<Dtype> > > layer_blobs;
                //for(int b = 0; b<sz; b++){
                //    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
                //    layer_blobs.push_back(blob_pointer);
                    //
                //    n=read_real<int>(is);
                //    c=read_real<int>(is);
                //    w=read_real<int>(is);
                //    h=read_real<int>(is);
                //    blob_pointer->Reshape(n, c, w, h);
                //    read_real<Dtype>(is, blob_pointer->mutable_cpu_data(), blob_pointer->count() );
                //}

				//read data and change
				vector<shared_ptr<Blob<Dtype> > > layer_blobs;
				blob_uchar2float(is, layer_blobs);
				
                //
                shared_ptr<InnerProductLayer<Dtype> > sl(new InnerProductLayer<Dtype>( ) );
                sl->CopyFrom(pt);
				sl->CopyBlob(layer_blobs);
                sl->SetUp(this->bottom_vecs()[i], this->top_vecs()[i]);
                
                layers_.push_back(sl);

				//LOGD("layer i:%d type:%s end...", i, layer_type.c_str() );
	        }
           
        }
        is.close();
        
        
    }


	template <typename Dtype>
	void  Net<Dtype>::blob_float2uchar(std::ostream &os, vector<shared_ptr<Blob<Dtype>>> &inBlobs)
	{
		//
		int j,k,len;
		Dtype tmp;

		//blob.size
		write_real<uint64_t>(os, inBlobs.size());
		    
		for(j = 0; j < inBlobs.size(); j++){
			//write_real<Dtype>(os, blobs[b]->mutable_cpu_data(), blobs[b]->count());

			len = inBlobs[j]->count();
			Dtype* blob_data = inBlobs[j]->mutable_cpu_data();
			Dtype* minmax = new Dtype[2];//0-min,1-max
			unsigned char* ucharBolbData = new unsigned char[len];
			memset(ucharBolbData,0,len);
			
			//get minmax
			minmax[0] = 1000000.0;
			minmax[1] = -1000000.0;
			for(k = 0; k < len; k++)
			{
				if ( blob_data[k] < minmax[0] )
					minmax[0] = blob_data[k];
				if ( blob_data[k] > minmax[1] )
					minmax[1] = blob_data[k];
			}

			//Normalization
			if ( minmax[0] != minmax[1] )
			{
				tmp = 255.0/(minmax[1] - minmax[0]);
				for(k = 0; k < len; k++)
				{
					ucharBolbData[k] = (unsigned char)((blob_data[k]-minmax[0])*tmp + 0.5) ;
				}
			}

			//write data
		    write_real<int>(os, inBlobs[j]->num());
		    write_real<int>(os, inBlobs[j]->channels());
		    write_real<int>(os, inBlobs[j]->width());
		    write_real<int>(os, inBlobs[j]->height());
		    write_real<Dtype>(os, minmax, 2);
			write_real<unsigned char>(os, ucharBolbData, len);

			//check
			std::cout<<"len:"<<len<<","<<inBlobs[j]->width()<<"x"<<inBlobs[j]->height()
		      <<"x"<<inBlobs[j]->channels()<<"x"<<inBlobs[j]->num()
		      <<",min:"<<minmax[0]<<",max:"<<minmax[1]<<std::endl;

			if (minmax) {delete [] minmax;minmax = 0;}
			if (ucharBolbData) {delete [] ucharBolbData;ucharBolbData = 0;}
		}

	}

	template <typename Dtype>
	void  Net<Dtype>::blob_uchar2float(std::istream &is, vector<shared_ptr<Blob<Dtype> > > &outBlobs)
	{
		//
		int j,k,n,c,w,h;
		uint64_t len,blob_size;
		Dtype tmp;

		//blob.size
		blob_size = read_real<uint64_t>(is);
		//printf("blob_size:%ld\n",blob_size);
		    
		for(j = 0; j < blob_size; j++){

			//read data
			shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
		    outBlobs.push_back(blob_pointer);

			n=read_real<int>(is);
		    c=read_real<int>(is);
		    w=read_real<int>(is);
		    h=read_real<int>(is);
			len = n*c*w*h;
		    blob_pointer->Reshape(n, c, w, h);
			std::cout<<"len:"<<len<<","<<w<<"x"<<h<<"x"<<c<<"x"<<n<<std::endl;
			
			
			Dtype* minmax = new Dtype[2];	//0-min,1-max
			unsigned char* ucharBolbData = new unsigned char[len];	//uchar
			Dtype* blob_data = new Dtype[len];	//float
			memset(minmax,0,2);
			memset(ucharBolbData,0,len);
			memset(blob_data,0,len);

			//read data
			read_real<Dtype>(is, minmax, 2);
			read_real<unsigned char>(is, ucharBolbData, len);
			std::cout<<"min:"<<minmax[0]<<",max:"<<minmax[1]<<std::endl;

			//Normalization
			tmp = (minmax[1] - minmax[0])/255.0;
			for(k = 0; k < len; k++)
			{
				blob_data[k] = (Dtype)( ucharBolbData[k]*tmp + minmax[0] ) ;
			}
			caffe_copy(len, blob_data, blob_pointer->mutable_cpu_data());

			//check
			std::cout<<"len:"<<len<<","<<w<<"x"<<h<<"x"<<c<<"x"<<n<<",min:"<<minmax[0]<<",max:"<<minmax[1]<<std::endl;
			
			if (minmax) {delete [] minmax;minmax = 0;}
			if (ucharBolbData) {delete [] ucharBolbData;ucharBolbData = 0;}
			if (blob_data) {delete [] blob_data;blob_data = 0;}

			
		}

	}
    

    
    INSTANTIATE_CLASS(Net);
    
}  // namespace caffe
