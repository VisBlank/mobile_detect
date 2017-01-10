#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
//#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

    typedef struct ConvolutionParam {
        bool force_nd_im2col;
        int channel_axis;
        int num_spatial_axes;
        Blob<int> kernel_shape;
        Blob<int> stride;
        Blob<int> pad;
        Blob<int> dilation;
        bool is_1x1;
        int channels;
        int num_output;
        int group;
        int conv_out_channels;
        int conv_in_channels;
        bool bias_term;
        int kernel_dim;
    }ConvolutionParam;
/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  /*explicit BaseConvolutionLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {
        //std::cout<<"conv layer init."<<std::endl;
        conv_param.CopyFrom(param.convolution_param());
    }*/
    
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

  //explicit BaseConvolutionLayer(const shared_ptr<Layer<Dtype> >& param): Layer<Dtype>(param) {}
    explicit BaseConvolutionLayer(): Layer<Dtype>() {}
    /*
    explicit BaseConvolutionLayer(const shared_ptr<Layer<Dtype> >& param): Layer<Dtype>(param)
    {
        //std::cout<<"conv layer copy:"<<param->type()<<std::endl;
        conv_param.CopyFrom(param->layer_param().convolution_param()) ;
        
    }
    void CopyFrom(const LayerParameter& layer_param)
    {
        //std::cout<<"conv param copy:"<<std::endl;
        conv_param.CopyFrom(layer_param.convolution_param()) ;
    }*/
    void CopyFrom(const ConvolutionParam& p)
    {
        
        force_nd_im2col_ = p.force_nd_im2col;
        channel_axis_ = p.channel_axis;
        num_spatial_axes_ = p.num_spatial_axes;
        kernel_shape_.CopyFrom(p.kernel_shape, false, true);
        stride_.CopyFrom(p.stride, false, true);
        pad_.CopyFrom(p.pad, false, true);
        dilation_.CopyFrom(p.dilation, false, true);
        is_1x1_ = p.is_1x1;
        channels_ = p.channels;
        num_output_ = p.num_output;
        group_ = p.group;
        conv_out_channels_ = p.conv_out_channels;
        conv_in_channels_ = p.conv_in_channels;
        bias_term_ = p.bias_term;
        kernel_dim_ = p.kernel_dim;
        
        
    }
    void CopyFrom(const BaseConvolutionLayer<Dtype>  *conv)
    {

        num_spatial_axes_ = conv->num_spatial_axes();
        bottom_dim_ =  conv->bottom_dim();
        top_dim_ = conv->top_dim();
        channel_axis_ = conv->channel_axis();
        num_ = conv->num();
        channels_ = conv->channels();
        group_ = conv->group();
        out_spatial_dim_ = conv->out_spatial_dim();
        weight_offset_ = conv->weight_offset();
        num_output_ = conv->num_output();
        bias_term_ = conv->bias_term();
        is_1x1_ = conv->is_1x1();
        force_nd_im2col_ = conv->force_nd_im2col();
        num_kernels_im2col_ = conv->num_kernels_im2col();
        num_kernels_col2im_ = conv->num_kernels_col2im();
        conv_out_channels_ = conv->conv_out_channels();
        conv_in_channels_ = conv->conv_in_channels();
        conv_out_spatial_dim_ = conv->conv_out_spatial_dim();
        kernel_dim_ = conv->kernel_dim();
        col_offset_ = conv->col_offset();
        output_offset_ = conv->output_offset();

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
        
        kernel_dim_ = this->blobs_[0]->count(1);
        weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
        // Propagate gradients to the parameters (as directed by backward pass).
        //this->param_propagate_down_.resize(this->blobs_.size(), true);
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

        kernel_dim_ = this->blobs_[0]->count(1);
        weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
        // Propagate gradients to the parameters (as directed by backward pass).
        //this->param_propagate_down_.resize(this->blobs_.size(), true);
    }

  //
  inline const int& num_spatial_axes() const {        return num_spatial_axes_;  }
  inline const int& bottom_dim() const {        return bottom_dim_;  }
  inline const int& top_dim() const {        return top_dim_;  }
  inline const int& channel_axis() const {        return channel_axis_;  }
  inline const int& num() const {        return num_;  }
  inline const int& channels() const {        return channels_;  }
  inline const int& group() const {        return group_;  }
  inline const int& out_spatial_dim() const {        return out_spatial_dim_;  }
  inline const int& weight_offset() const {        return weight_offset_;  }
  inline const int& num_output() const {        return num_output_;  }
  inline const bool& bias_term() const {        return bias_term_;  }
  inline const bool& is_1x1() const {        return is_1x1_;  }
  inline const bool& force_nd_im2col() const {        return force_nd_im2col_;  }
  inline const int& num_kernels_im2col() const {        return num_kernels_im2col_;  }
  inline const int& num_kernels_col2im() const {        return num_kernels_col2im_;  }
  inline const int& conv_out_channels() const {        return conv_out_channels_;  }
  inline const int& conv_in_channels() const {        return conv_in_channels_;  }
  inline const int& conv_out_spatial_dim() const {        return conv_out_spatial_dim_;  }
  inline const int& kernel_dim() const {        return kernel_dim_;  }
  inline const int& col_offset() const {        return col_offset_;  }
  inline const int& output_offset() const {        return output_offset_;  }
    
  inline const  Blob<int>& kernel_shape() const { return kernel_shape_;}
  inline const  Blob<int>& stride() const { return stride_;}
  inline const  Blob<int>& pad() const { return pad_;}
  inline const  Blob<int>& dilation() const { return dilation_;}
  inline const  Blob<int>& conv_input_shape() const { return conv_input_shape_;}
  inline const  vector<int>& col_buffer_shape() const { return col_buffer_shape_;}
  inline const  vector<int>& output_shape() const { return output_shape_;}
  inline const  Blob<Dtype>& col_buffer() const { return col_buffer_;}
  inline const  Blob<Dtype>& bias_multiplier() const { return bias_multiplier_;}
    
 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  //ConvolutionParameter conv_param;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
