#include <vector>

#include "caffe/layers/abs_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AbsLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void AbsLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  // do difference and store for backprob
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  
  Dtype loss = caffe_cpu_asum(count, diff_.cpu_data()) / bottom[0]->num();


  //Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  //Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss ;
  
}

/* GOOD to know comment Ravi
For loss layers, there is no next layer, and so the top diff blob is technically undefined and unused - but Caffe is using this preallocated space to store unrelated data: Caffe supports multiplying loss layers with a user-defined weight (loss_weight in the prototxt), this information (a single scalar floating point number) is stored in the first element of the diff array of the top blob. That's why you'll see in every loss layer, that they multiply by that amount to support that functionality. This is explained in Caffe's tutorial about the loss layer.

This weight is usually used to add auxiliary losses to the network. You can read more about it in Google's Going Deeper with Convoltions or in Deeply-Supervised Nets.
*/

template <typename Dtype>
void AbsLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
        int count = bottom[i]->count();
     	const Dtype sign = (i == 0) ? 1 : -1;
	const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
        const Dtype* diff = diff_.cpu_data();
        for (int i = 0; i < count; ++i) {
           bottom_diff[i] = alpha * (diff[i] > 0) - alpha * (diff[i] <= 0);
        }
        
/*        caffe_cpu_sign(count, diff_.cpu_data(), bottom[i]->mutable_cpu_diff());
	
        caffe_scal(
          count,
	  alpha,
          bottom[i]->mutable_gpu_diff(),
          1);

        caffe_cpu_axpby(
          count,              // count
          alpha,                              // alpha
          bottom[i]->mutable_cpu_diff(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b

	
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    */
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AbsLossLayer);
#endif

INSTANTIATE_CLASS(AbsLossLayer);
REGISTER_LAYER_CLASS(AbsLoss);

}  // namespace caffe
