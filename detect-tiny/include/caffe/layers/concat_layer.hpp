#ifndef CAFFE_CONCAT_LAYER_HPP_
#define CAFFE_CONCAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
//#include "caffe/proto/caffe.pb.h"

namespace caffe {

    typedef struct ConcatParam{
        int concat_axis;
    }ConcatParam;
/**
 * @brief Takes at least two Blob%s and concatenates them along either the num
 *        or channel dimension, outputting the result.
 */
template <typename Dtype>
class ConcatLayer : public Layer<Dtype> {
 public:
  //explicit ConcatLayer(const LayerParameter& param)
  //    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Concat"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  //explicit ConcatLayer(const shared_ptr<Layer<Dtype> >& param): Layer<Dtype>(param) {}
  explicit ConcatLayer(): Layer<Dtype>() {}
  void CopyFrom(const ConcatParam& p)
  {
      concat_axis_ = p.concat_axis;
  }
  //
  inline const int& count() const {        return count_;  }
  inline const int& num_concats() const {        return num_concats_;  }
  inline const int& concat_input_size() const {        return concat_input_size_;  }
  inline const int& concat_axis() const {        return concat_axis_;  }
    
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);



  int count_;
  int num_concats_;
  int concat_input_size_;
  int concat_axis_;
};

}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
