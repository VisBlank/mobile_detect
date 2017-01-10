// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#ifndef CAFFE_FAST_RCNN_LAYERS_HPP_
#define CAFFE_FAST_RCNN_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
//#include "caffe/proto/caffe.pb.h"

//#include "caffe/layers/loss_layer.hpp"

namespace caffe {

  //
  typedef struct ROIPoolingParam {
  		int pooled_width_;
        int pooled_height_;
        float spatial_scale_;
  }ROIPoolingParam;


/* ROIPoolingLayer - Region of Interest Pooling Layer
*/
template <typename Dtype>
class ROIPoolingLayer : public Layer<Dtype> {
 
 public:
  //explicit ROIPoolingLayer(const LayerParameter& param)
  //    : Layer<Dtype>(param) {
  //    roi_pool_param.CopyFrom(param.roi_pooling_param()) ;
  //}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIPooling"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

  //add by chigo 20170105
	explicit ROIPoolingLayer(): Layer<Dtype>() {}
	//explicit ROIPoolingLayer(const shared_ptr<Layer<Dtype> >& param): Layer<Dtype>(param) {
	//	std::cout<<"roi pooling layer copy:"<<param->type()<<std::endl;
	//	roi_pool_param.CopyFrom(param->layer_param().roi_pooling_param()) ;
	//}

	//void CopyFrom(const LayerParameter& layer_param)
	//{
	//	//std::cout<<"pooling param copy:"<<std::endl;
	//	roi_pool_param.CopyFrom(layer_param.roi_pooling_param()) ;
	//}

	void CopyFrom(const ROIPoolingParam& p)
	{      
		pooled_width_ = p.pooled_width_;
		pooled_height_  = p.pooled_height_;
		spatial_scale_ = (Dtype)p.spatial_scale_;
	}
	
	void CopyFrom(const shared_ptr<ROIPoolingLayer<Dtype> >& pl)
	{
		channels_ = pl->channels();
		height_ = pl->height();
        width_ = pl->width();
        pooled_height_ = pl->pooled_height();
		pooled_width_ = pl->pooled_width();
		spatial_scale_ = pl->spatial_scale();
        max_idx_.ReshapeLike(pl->max_idx());
	}
  
    void CopyFrom(const ROIPoolingLayer<Dtype>  *pl)
    {
    	channels_ = pl->channels();
		height_ = pl->height();
        width_ = pl->width();
        pooled_height_ = pl->pooled_height();
		pooled_width_ = pl->pooled_width();
		spatial_scale_ = pl->spatial_scale();
        max_idx_.ReshapeLike(pl->max_idx());
    }

	inline const int& channels() const { return channels_;}
	inline const int& height() const { return height_; }
	inline const int& width() const { return width_; }
	inline const int& pooled_height() const { return pooled_height_;}
	inline const int& pooled_width() const { return pooled_width_; }
	inline const Dtype& spatial_scale() const { return spatial_scale_; }
	inline const Blob<int>& max_idx() const { return max_idx_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  Dtype spatial_scale_;
  Blob<int> max_idx_;
  //ROIPoolingParameter roi_pool_param ;

};


/*
template <typename Dtype>
class SmoothL1LossLayer : public LossLayer<Dtype> {
 public:
  explicit SmoothL1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothL1Loss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 4; }

  //Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
  //to both inputs -- override to return true and always allow force_backward.
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> errors_;
  Blob<Dtype> ones_;
  bool has_weights_;
  Dtype sigma2_;
};
*/



  //
  typedef struct ProposalParam {
	int base_size_;
	int feat_stride_;
	int pre_nms_topn_;
	int post_nms_topn_;
	float nms_thresh_;
	int min_size_;
	vector<float> ratios_;
	vector<float> scales_;
  }ProposalParam;

template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
  
 public:
  //explicit ProposalLayer(const LayerParameter& param)
  //    : Layer<Dtype>(param) {
  //    proposal_param.CopyFrom(param.proposal_param()) ;
  //}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //LOG(FATAL) << "Reshaping happens during the call to forward.";
  }

  virtual inline const char* type() const { return "ProposalLayer"; }

  //add by chigo 20170105
	explicit ProposalLayer(): Layer<Dtype>() {}
	//explicit ProposalLayer(const shared_ptr<Layer<Dtype> >& param): Layer<Dtype>(param) {
	//	std::cout<<"proposal layer copy:"<<param->type()<<std::endl;
	//	proposal_param.CopyFrom(param->layer_param().proposal_param()) ;
	//}

	//void CopyFrom(const LayerParameter& layer_param)
	//{
	//	//std::cout<<"pooling param copy:"<<std::endl;
	//	proposal_param.CopyFrom(layer_param.proposal_param()) ;
	//}

	void CopyFrom(const ProposalParam& p)
	{      
		base_size_ = p.base_size_;
		feat_stride_ = p.feat_stride_;
        pre_nms_topn_ = p.pre_nms_topn_;
        post_nms_topn_ = p.post_nms_topn_;
		nms_thresh_ = p.nms_thresh_;
		min_size_ = p.min_size_;
		ratios_ = p.ratios_;
		scales_ = p.scales_;
	}
	
	void CopyFrom(const shared_ptr<ProposalLayer<Dtype> >& pl)
	{
		base_size_ = pl->base_size();
		feat_stride_ = pl->feat_stride();
        pre_nms_topn_ = pl->pre_nms_topn();
        post_nms_topn_ = pl->post_nms_topn();
		nms_thresh_ = pl->nms_thresh();
		min_size_ = pl->min_size();
        anchors_.ReshapeLike(pl->anchors());
		proposals_.ReshapeLike(pl->proposals());
		roi_indices_.ReshapeLike(pl->roi_indices());
		nms_mask_.ReshapeLike(pl->nms_mask());
	}
  
    void CopyFrom(const ProposalLayer<Dtype>  *pl)
    {
    	base_size_ = pl->base_size();
		feat_stride_ = pl->feat_stride();
        pre_nms_topn_ = pl->pre_nms_topn();
        post_nms_topn_ = pl->post_nms_topn();
		nms_thresh_ = pl->nms_thresh();
		min_size_ = pl->min_size();
        anchors_.ReshapeLike(pl->anchors());
		proposals_.ReshapeLike(pl->proposals());
		roi_indices_.ReshapeLike(pl->roi_indices());
		nms_mask_.ReshapeLike(pl->nms_mask());
    }

	inline const int& base_size() const { return base_size_;}
	inline const int& feat_stride() const { return feat_stride_; }
	inline const int& pre_nms_topn() const { return pre_nms_topn_; }
	inline const int& post_nms_topn() const { return post_nms_topn_;}
	inline const Dtype& nms_thresh() const { return nms_thresh_; }
	inline const int& min_size() const { return min_size_; }
	inline const Blob<Dtype>& anchors() const { return anchors_; }
	inline const Blob<Dtype>& proposals() const { return proposals_; }
	inline const Blob<int>& roi_indices() const { return roi_indices_; }
	inline const Blob<int>& nms_mask() const { return nms_mask_; }

	inline const vector<Dtype>& vecRatios() const { return ratios_; }
	inline const vector<Dtype>& vecScales() const { return scales_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  int base_size_;
  int feat_stride_;
  int pre_nms_topn_;
  int post_nms_topn_;
  Dtype nms_thresh_;
  int min_size_;
  Blob<Dtype> anchors_;
  Blob<Dtype> proposals_;
  Blob<int> roi_indices_;
  Blob<int> nms_mask_;

  //add by chigo
  //ProposalParameter proposal_param;
  vector<Dtype> ratios_;
  vector<Dtype> scales_;
  
};

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_LAYERS_HPP_
