#include <cfloat>
#include <vector>

#include "caffe/layers/pin_hole_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void PinHoleProjection(const int nthreads, const int H, const int W, const int bottom_channel,const int top_channel,  const Dtype* const pts3D, const Dtype* const camIntrinsic, Dtype* const flows, Dtype* const proj_coords) 
{

  // pts3D (shape: N,3,H,W)
  // flows: (shape: N,2,H,W)
  // proj_coords: (shape: N,2,H,W)
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    const int x = index % W;
    const int y = (index / W) % H;
    const int n = index / W / H;
    // const int bottom_channel = 4;
    // const int top_channel = 2;
    const int X_offset = ((n * bottom_channel + 0) * H + y) * W + x;
    const int Y_offset = ((n * bottom_channel + 1) * H + y) * W + x;
    const int Z_offset = ((n * bottom_channel + 2) * H + y) * W + x;

    const Dtype* const camIntrin_off = camIntrinsic + n*4; 


    const float fx = camIntrin_off[0];
    const float fy = camIntrin_off[1];
    const float cx = camIntrin_off[2];
    const float cy = camIntrin_off[3];

    const Dtype X = pts3D[X_offset];
    const Dtype Y = pts3D[Y_offset];
    const Dtype Z = pts3D[Z_offset];


    int x_offset = ((n * top_channel + 0) * H + y) * W + x;
    int y_offset = ((n * top_channel + 1) * H + y) * W + x;

    flows[x_offset] = fx * X / (Z+1e-12) + cx - x;
    flows[y_offset] = fy * Y / (Z+1e-12) + cy - y;

    proj_coords[x_offset] = fx * X / (Z+1e-12) + cx ;
    proj_coords[y_offset] = fy * Y / (Z+1e-12) + cy ;
   }
}

template <typename Dtype>
void PinHoleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int bottom_channel = bottom[0]->channels();
  int top_channel = top[0]->channels();
  int width = bottom[0]->width();
  int n_threads = num * height * width;

  // bottom[0] --> 3D points (N,C,H,W)
  // bottom[1] --> camera intrinsic (N,4,1,1)
  // top[0]    --> flows (N,2,H,W) [u,v]
  // top[1]    --> projection coordinat (N,2,H,W) [xp,yp]

  PinHoleProjection <<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
                    n_threads, height, width,bottom_channel, top_channel, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
                    top[0]->mutable_gpu_data(), top[1]->mutable_gpu_data());
}



template <typename Dtype>
__global__ void GetGradient(const int nthreads,
    const Dtype* const top_flows_diff, const Dtype* const top_coords_diff, const Dtype* const pts3D, const Dtype* const camIntrinsic, 
                  const int height, const int width, const int bottom_channel, const int top_channel,Dtype* const pts3D_diff, Dtype* const camIntrinsic_diff){

    CUDA_KERNEL_LOOP(index, nthreads) {
    const int x = index % width;
    const int y = (index / width) % height;
    // const int bottom_channel = 4;
    // const int top_channel = 2;
    const int n = index / width / height ;

    const Dtype* const camIntrin_off = camIntrinsic + n*4; 
    const int X_offset = ((n * bottom_channel + 0) * height + y) * width + x;
    const int Y_offset = ((n * bottom_channel + 1) * height + y) * width + x;
    const int Z_offset = ((n * bottom_channel + 2) * height + y) * width + x;

    const int x_offset = ((n * top_channel + 0) * height + y) * width + x;
    const int y_offset = ((n * top_channel + 1) * height + y) * width + x;

    const float fx = camIntrin_off[0];
    const float fy = camIntrin_off[1];
    // const float cx = camIntrin_off[2];
    // const float cy = camIntrin_off[3];

    // Setup base grid
    const Dtype X = pts3D[X_offset];
    const Dtype Y = pts3D[Y_offset];
    const Dtype Z = pts3D[Z_offset];
 
    // bottom[0] dLd[X,Y,Z]
    Dtype X_diff_val(0);
    X_diff_val += top_flows_diff[x_offset] * fx / (Z+1e-12);
    X_diff_val += top_coords_diff[x_offset] * fx / (Z+1e-12);
    pts3D_diff[X_offset] = X_diff_val;

    Dtype Y_diff_val(0);
    Y_diff_val += top_flows_diff[y_offset] * fy / (Z+1e-12);
    Y_diff_val += top_coords_diff[y_offset] * fy / (Z+1e-12);
    pts3D_diff[Y_offset] = Y_diff_val;

    Dtype Z_diff_val(0);
    Z_diff_val += - top_flows_diff[x_offset] * fx * X / (Z*Z+1e-12);
    Z_diff_val += - top_coords_diff[x_offset] * fx * X / (Z*Z+1e-12);
    Z_diff_val += - top_flows_diff[y_offset] * fy * Y / (Z*Z+1e-12);
    Z_diff_val += - top_coords_diff[y_offset] * fy * Y / (Z*Z+1e-12);
    pts3D_diff[Z_offset] = Z_diff_val;

    // bootom[1] dLdK
    Dtype cx_diff_val(0); 
    cx_diff_val += top_flows_diff[x_offset];
    cx_diff_val += top_coords_diff[x_offset];
    caffe_gpu_atomic_add( cx_diff_val, camIntrinsic_diff + n*4 + 2); 

    Dtype cy_diff_val(0); 
    cy_diff_val += top_flows_diff[y_offset];
    cy_diff_val += top_coords_diff[y_offset];
    caffe_gpu_atomic_add( cy_diff_val, camIntrinsic_diff + n*4 + 3); 

    Dtype fx_diff_val(0); 
    fx_diff_val += top_flows_diff[x_offset] * X / (Z+1e-12);
    fx_diff_val += top_coords_diff[x_offset] * X / (Z+1e-12);
    caffe_gpu_atomic_add( fx_diff_val, camIntrinsic_diff + n*4 + 0);

    Dtype fy_diff_val(0); 
    fy_diff_val += top_flows_diff[y_offset] * Y/ (Z+1e-12);
    fy_diff_val += top_coords_diff[y_offset] * Y / (Z+1e-12);
    caffe_gpu_atomic_add( fy_diff_val, camIntrinsic_diff + n*4 + 1);

  }
}

template <typename Dtype>
void PinHoleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{  
  const Dtype* top_flows_diff = top[0]->gpu_diff();
  const Dtype* top_coords_diff = top[1]->gpu_diff();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int bottom_channel = bottom[0]->channels();
  int top_channel = top[0]->channels();
  int n_threads = num * height * width;

  const Dtype* pts3D = bottom[0]->gpu_data();
  const Dtype* camIntrinsic = bottom[1]->gpu_data();

  Dtype* pts3D_diff = bottom[0]->mutable_gpu_diff();
  Dtype* camIntrinsic_diff = bottom[1]->mutable_gpu_diff();

  caffe_gpu_set(bottom[0]->count(), (Dtype)0., pts3D_diff);
  caffe_gpu_set(bottom[1]->count(), (Dtype)0., camIntrinsic_diff);

  GetGradient<Dtype><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
                        n_threads, top_flows_diff, top_coords_diff, pts3D, camIntrinsic, 
                  height, width, bottom_channel, top_channel, pts3D_diff, camIntrinsic_diff);
  CUDA_POST_KERNEL_CHECK; 
}

INSTANTIATE_LAYER_GPU_FUNCS(PinHoleLayer);

}  // namespace caffe
