#include <vector>

#include "caffe/layers/abs_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



template <typename Dtype>
void AbsLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype loss;
  caffe_gpu_asum(count, diff_.gpu_data(), &loss) ;
  loss = loss / bottom[0]->num();
  //Dtype dot;
  //caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  //Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void AbsLossBackward(const int n,
    const Dtype* diff_data, Dtype* out_diff, const Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = alpha * ( (diff_data[index] > 0) - (diff_data[index] <= 0));
  }
}

template <typename Dtype>
void AbsLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      int count = bottom[i]->count();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();

      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const Dtype* diff = diff_.gpu_data();
      //ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      //  count, top_diff, bottom_data, bottom_diff, negative_slope);
      AbsLossBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, diff, bottom_diff, alpha);
      CUDA_POST_KERNEL_CHECK;
    /*for (int i = 0; i < count; ++i) {
           bottom_diff[i] = alpha * (diff[i] > 0) - alpha * (diff[i] < 0);
        }*/

/*     caffe_gpu_sign(count, diff_.gpu_data(), bottom[i]->mutable_gpu_diff());
      caffe_gpu_scal(
          count,
	  alpha,
          bottom[i]->mutable_gpu_diff(),
          1);
       caffe_gpu_axpby(
          count,              // count
          alpha,                              // alpha
          bottom[i]->mutable_gpu_diff(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b



      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
*/
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AbsLossLayer);

}  // namespace caffe
