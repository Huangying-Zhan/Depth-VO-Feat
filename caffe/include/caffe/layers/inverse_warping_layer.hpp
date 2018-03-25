#ifndef CAFFE_INVERSE_WARPING_LAYER_HPP_
#define CAFFE_INVERSE_WARPING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Compute forward warp image given,
 *        input image, projection coordinate
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template <typename Dtype>
class InverseWarpingLayer : public Layer<Dtype> {
 public:
  explicit InverseWarpingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // InverseWarping layer outputs a warp image given an input image and projection coordinate;
  
  virtual inline const char* type() const { return "InverseWarping";}
  // virtual inline int MinBottomBlobs() const { return 3; }
  //virtual inline int ExactNumTopBlobs() const { return 1; } 
  // If 2 blobs at top the external occussion mask is returned
  //virtual inline int MinTopBlobs() const { return 1; }
  // virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //int kernel_h_, kernel_w_;
  //int stride_h_, stride_w_;
  //int pad_h_, pad_w_;
  // int channels_;
  // int height_, width_;
  
  // to use top_mask
  // bool mask_flag ;  
  // Blob<Dtype> ext_occ;
  //int ext_occ_panelty = 0.01;


  //Blob<Dtype> warp_image;

  /*int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;*/
};

}  // namespace caffe

#endif  // CAFFE_WARPING_LAYER_HPP_
