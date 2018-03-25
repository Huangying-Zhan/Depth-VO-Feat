#ifndef CAFFE_GEOTRANSFORM_LAYER_HPP_
#define CAFFE_GEOTRANSFORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GeoTransformLayer : public Layer<Dtype> {
 public:
  explicit GeoTransformLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GeoTransform"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index == 0;
  }
  virtual inline int ExactNumBottomBlobs() const { return -1; } 
  virtual inline int ExactNumTopBlobs() const { return -1; } 
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_GEOTRANSFORM_LAYER_HPP_
