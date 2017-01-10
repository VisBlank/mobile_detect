#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
//#include "caffe/proto/caffe.pb.h"

namespace caffe {

  //
  typedef struct InnerProductParam {
	bool bias_term_;
	bool transpose_;
    int num_output_;
    int axis_;
  }InnerProductParam;

/*  enum {
	CblasNoTrans_ = 111, 
	CblasTrans_ = 112, 
	CblasConjTrans_ = 113
  } CBLAS_TRANSPOSE;
*/

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {

 public:
  //explicit InnerProductLayer(const LayerParameter& param)
  //    : Layer<Dtype>(param) {
  //    tParam.CopyFrom(param.inner_product_param()) ;
  //}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  //add by chigo 20170105
	explicit InnerProductLayer(): Layer<Dtype>() {}
	//explicit InnerProductLayer(const shared_ptr<Layer<Dtype> >& param): Layer<Dtype>(param) {
	//	//std::cout<<"InnerProduct layer copy:"<<param->type()<<std::endl;
	//	tParam.CopyFrom(param->layer_param().inner_product_param()) ;
	//}

	//void CopyFrom(const LayerParameter& layer_param)
	//{
	//	//std::cout<<"InnerProduct param copy:"<<std::endl;
	//	tParam.CopyFrom(layer_param.inner_product_param()) ;
	//}

	void CopyFrom(const InnerProductParam& p)
	{      
		bias_term_ = p.bias_term_;
		transpose_ = p.transpose_;
        num_output_ = p.num_output_;
        axis_ = p.axis_;
	}
	
	void CopyFrom(const shared_ptr<InnerProductLayer<Dtype> >& pl)
	{
		M_ = pl->M();
		K_ = pl->K();
		N_ = pl->N();
		bias_term_ = pl->bias_term();
		bias_multiplier_.ReshapeLike(pl->bias_multiplier());
		transpose_ = pl->transpose();
	}
  
    void CopyFrom(const InnerProductLayer<Dtype>  *pl)
    {
    	M_ = pl->M();
		K_ = pl->K();
		N_ = pl->N();
		bias_term_ = pl->bias_term();
		bias_multiplier_.ReshapeLike(pl->bias_multiplier());
		transpose_ = pl->transpose();
    }

	void CopyBlob(const  vector<shared_ptr<Blob<Dtype> > > & layer_blobs)
    {
        if(layer_blobs.size() > 0){
            //std::cout<<"copy blob."<<std::endl;
            this->blobs_.resize(layer_blobs.size());
            for(int b = 0; b < layer_blobs.size(); b++){
                this->blobs_[b].reset(new Blob<Dtype>());
                //this->blobs_[b]->CopyFrom(layer_blobs[b], false, true);
                this->blobs_[b]->Reshape(layer_blobs[b]->shape());
                caffe_copy(this->blobs_[b]->count(), layer_blobs[b]->cpu_data(), static_cast<Dtype*>(this->blobs_[b]->mutable_cpu_data()));
                
            }
        }
        
        //kernel_dim_ = this->blobs_[0]->count(1);
        //weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    }
	
    void CopyBlob(const shared_ptr<Layer<Dtype> >& other)
    {
        if(other->blobs().size() > 0){
            //std::cout<<"copy blob."<<std::endl;
            this->blobs_.resize(other->blobs().size());
            for(int b = 0; b < other->blobs().size(); b++){
                this->blobs_[b].reset(new Blob<Dtype>());
                //blobs_[b]->CopyFrom(other->blobs()[b]);
                this->blobs_[b]->Reshape(other->blobs()[b]->shape());
                //caffe_copy(this->blobs_[b]->count(), this->blobs_[b]->mutable_cpu_data(),other->blobs()[b]->mutable_cpu_data());
                caffe_copy(this->blobs_[b]->count(), other->blobs()[b]->cpu_data(), static_cast<Dtype*>(this->blobs_[b]->mutable_cpu_data()));

            }
        }

        //kernel_dim_ = this->blobs_[0]->count(1);
        //weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    }

	inline const int& M() const { return M_;}
	inline const int& K() const { return K_; }
	inline const int& N() const { return N_; }
	inline const bool& bias_term() const { return bias_term_; }
	inline const Blob<Dtype>& bias_multiplier() const { return bias_multiplier_; }
	inline const bool& transpose() const { return transpose_; }

	inline const int& num_output() const { return num_output_;}
	inline const int& axis() const { return axis_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

  //add
  //InnerProductParameter tParam;
  int num_output_;
  int axis_;
  
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
