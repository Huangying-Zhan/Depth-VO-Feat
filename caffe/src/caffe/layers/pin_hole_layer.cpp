#include <vector>

#include "caffe/layers/pin_hole_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe 
{

template <typename Dtype>
void PinHoleLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  top[0]->Reshape(bottom[0]->num(), 2, bottom[0]->height(), bottom[0]->width());
  top[1]->Reshape(bottom[0]->num(), 2, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void PinHoleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  std::cout << "PinHoleLayer Warning: Running Forward_gpu() code instead" << std::endl;

  // Forward_gpu(bottom, top);
}

template <typename Dtype>
void PinHoleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  std::cout << "PinHoleLayer Warning: Running Backward_gpu() code instead" << std::endl;

  // Backward_gpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(PinHoleLayer);
#endif

INSTANTIATE_CLASS(PinHoleLayer);
REGISTER_LAYER_CLASS(PinHole);

}  // namespace caffe
