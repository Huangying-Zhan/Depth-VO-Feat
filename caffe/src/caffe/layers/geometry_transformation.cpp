#include <vector>

#include "caffe/layers/geometry_transformation.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

// #define LEVELS 256 // cost volume levels
// #define MINID 1e-5
// #define MAXID 4

namespace caffe 
{

template <typename Dtype>
void GeoTransformLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  top[0]->Reshape(bottom[0]->num(), 3, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void GeoTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  std::cout << "GeoTransformLayer Warning: Running Forward_gpu() code instead" << std::endl;

  // Forward_gpu(bottom, top);
}

template <typename Dtype>
void GeoTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  std::cout << "GeoTransformLayer Warning: Running Backward_gpu() code instead" << std::endl;

  // Backward_gpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(GeoTransformLayer);
#endif

INSTANTIATE_CLASS(GeoTransformLayer);
REGISTER_LAYER_CLASS(GeoTransform);

}  // namespace caffe
