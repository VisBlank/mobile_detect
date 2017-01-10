#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
//#include "caffe/proto/caffe.pb.h"

namespace caffe {

    //
    typedef struct PoolingParam {
        int kernel_h, kernel_w;
        int stride_h, stride_w;
        int pad_h, pad_w;
        bool global_pooling;
        int pool_method;
    }PoolingParam;
    
    enum PoolingParam_PoolMethod {
        PoolingParameter_PoolMethod_MAX_ = 0,
        PoolingParameter_PoolMethod_AVE_ = 1,
        PoolingParameter_PoolMethod_STOCHASTIC_ = 2
    };

    
/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  
  /*explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
          //std::cout<<"pooling layer init."<<std::endl;
          pool_param.CopyFrom(param.pooling_param()) ;
      }*/
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "Pooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    //return (this->layer_param_.pooling_param().pool() ==
    //        PoolingParameter_PoolMethod_MAX) ? 2 : 1;
      return (pool_method_ ==
              PoolingParameter_PoolMethod_MAX_) ? 2 : 1;
  }

    explicit PoolingLayer(): Layer<Dtype>() {}
  //explicit PoolingLayer(const shared_ptr<Layer<Dtype> >& param): Layer<Dtype>(param)
  //{
      //std::cout<<"pooling layer copy:"<<param->type()<<std::endl;
      //pool_param.CopyFrom(param->layer_param().pooling_param()) ;

  //}
  //void CopyFrom(const LayerParameter& layer_param)
  //{
      //std::cout<<"pooling param copy:"<<std::endl;
      //pool_param.CopyFrom(layer_param.pooling_param()) ;
  //}
  void CopyFrom(const PoolingParam& p)
  {
      
      global_pooling_ = p.global_pooling;
      pool_method_  = p.pool_method;
      kernel_h_ = p.kernel_h;
      kernel_w_ = p.kernel_w;
      stride_h_ = p.stride_h;
      stride_w_ = p.stride_w;
      pad_h_ = p.pad_h;
      pad_w_ = p.pad_w;
      
  }
    
  void CopyFrom(const shared_ptr<PoolingLayer<Dtype> >& pl)
  {
      global_pooling_ = pl->global_pooling();
      kernel_h_ = pl->kernel_h();
      kernel_w_ = pl->kernel_w();
      stride_h_ = pl->stride_h();
      stride_w_ = pl->stride_w();
      pad_h_ = pl->pad_h();
      pad_w_ = pl->pad_w();
      height_ = pl->height();
      width_ = pl->width();
      pooled_width_ = pl->pooled_width();
      pooled_height_ = pl->pooled_height();
      channels_ = pl->channels();
      rand_idx_.ReshapeLike(pl->rand_idx());
      max_idx_.ReshapeLike(pl->max_idx());
  }
  
    void CopyFrom(const PoolingLayer<Dtype>  *pl)
    {
        kernel_h_ = pl->kernel_h();
        kernel_w_ = pl->kernel_w();
        stride_h_ = pl->stride_h();
        stride_w_ = pl->stride_w();
        pad_h_ = pl->pad_h();
        pad_w_ = pl->pad_w();
        height_ = pl->height();
        width_ = pl->width();
        pooled_width_ = pl->pooled_width();
        pooled_height_ = pl->pooled_height();
        channels_ = pl->channels();
        global_pooling_ = pl->global_pooling();
        rand_idx_.ReshapeLike(pl->rand_idx());
        max_idx_.ReshapeLike(pl->max_idx());
    }
    
  //
  inline const Blob<Dtype>& rand_idx() const {        return rand_idx_;  }
  inline const Blob<int>& max_idx() const {        return max_idx_;  }
  inline const int& kernel_h() const {        return kernel_h_;  }
  inline const int& kernel_w() const {        return kernel_w_;  }
  inline const int& stride_h() const {        return stride_h_;  }
  inline const int& stride_w() const {        return stride_w_;  }
  inline const int& pad_h() const {        return pad_h_;  }
  inline const int& pad_w() const {        return pad_w_;  }
  inline const int& height() const {        return height_;  }
  inline const int& width() const {        return width_;  }
  inline const int& pooled_height() const {        return pooled_height_;  }
  inline const int& pooled_width() const {        return pooled_width_;  }
  inline const int& channels() const {        return channels_;  }
  inline const bool& global_pooling() const {        return global_pooling_;  }
    inline const int & pool_method() const { return pool_method_; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;
  //
  //PoolingParameter pool_param ;
  int pool_method_;
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
