#ifndef CAFFE_XXX_LAYER_HPP_
#define CAFFE_XXX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
//#include "caffe/proto/caffe.pb.h"

namespace caffe {

  //
  typedef struct ReshapeParam {
  		vector<int> shape_;
        int axis_;
        int num_axes_;
  }ReshapeParam;

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
template <typename Dtype>
class ReshapeLayer : public Layer<Dtype> {
 public:
  //explicit ReshapeLayer(const LayerParameter& param)
  //    : Layer<Dtype>(param) {
  //    t_reshape_param.CopyFrom(param.reshape_param()) ;
  //}
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reshape"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

	//add by chigo 20170105
	explicit ReshapeLayer(): Layer<Dtype>() {}
	//explicit ReshapeLayer(const shared_ptr<Layer<Dtype> >& param): Layer<Dtype>(param) {
	//	std::cout<<"reshape layer copy:"<<param->type()<<std::endl;
	//	t_reshape_param.CopyFrom(param->layer_param().reshape_param()) ;
	//}

	//void CopyFrom(const LayerParameter& layer_param)
	//{
	//	//std::cout<<"pooling param copy:"<<std::endl;
	//	t_reshape_param.CopyFrom(layer_param.reshape_param()) ;
	//}
	
	void CopyFrom(const ReshapeParam& p)
	{      
		shape_ = p.shape_;
		axis_  = p.axis_;
		num_axes_ = p.num_axes_;
	}
	
	void CopyFrom(const shared_ptr<ReshapeLayer<Dtype> >& pl)
	{
		copy_axes_ = pl->copy_axes();
		inferred_axis_ = pl->inferred_axis();
		constant_count_ = pl->constant_count();
	}
  
    void CopyFrom(const ReshapeLayer<Dtype>  *pl)
    {
    	copy_axes_ = pl->copy_axes();
		inferred_axis_ = pl->inferred_axis();
		constant_count_ = pl->constant_count();
    }

	inline const vector<int>& copy_axes() const { return copy_axes_;}
	inline const int& inferred_axis() const { return inferred_axis_; }
	inline const int& constant_count() const { return constant_count_; }
	
	inline const vector<int>& shape() const { return shape_;}
	inline const int& axis() const { return axis_; }
	inline const int& num_axes() const { return num_axes_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  //virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  /// @brief vector of axes indices whose dimensions we'll copy from the bottom
  vector<int> copy_axes_;
  /// @brief the index of the axis whose dimension we infer, or -1 if none
  int inferred_axis_;
  /// @brief the product of the "constant" output dimensions
  int constant_count_;

  //add by chigo 20170105
  //ReshapeParameter t_reshape_param;
  vector<int> shape_;
  int axis_;
  int num_axes_;

};

}  // namespace caffe

#endif  // CAFFE_XXX_LAYER_HPP_
