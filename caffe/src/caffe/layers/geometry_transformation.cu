#include <cfloat>
#include <vector>

#include "caffe/layers/geometry_transformation.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void GeoTransform(const int nthreads, const int H, const int W, const int top_channel, const Dtype* const depths, const Dtype* const transformation, const Dtype* const camIntrinsic, Dtype* const transformed_points) 
{

  // transformation (shape: N,1,4,4)
  // FIXME
  // camIntrinsic: (shape: N,4,1,1; fx, fy, cx, cy)
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    const int x = index % W;
    const int y = (index / W) % H;
    const int n = index / W / H;
    const int offset = (n * H + y) * W + x;
    const Dtype* const trans_off = transformation + n*16;
    const Dtype* const camIntrin_off = camIntrinsic + n*4;

    const float fx = camIntrin_off[0];
    const float fy = camIntrin_off[1];
    const float cx = camIntrin_off[2];
    const float cy = camIntrin_off[3];


    const Dtype d = depths[offset];
    // fixme
    // Dtype disp = depths[offset];
    // Dtype d = fx*0.54/(disp+1e-3);
    // fixme
    const Dtype X = (x-cx)/fx*d;
    const Dtype Y = (y-cy)/fy*d;

    int x_offset = ((n * top_channel + 0) * H + y) * W + x;
    transformed_points[x_offset] = trans_off[0]*X + trans_off[1]*Y + trans_off[2]*d + trans_off[3];
    int y_offset = ((n * top_channel + 1) * H + y) * W + x;
    transformed_points[y_offset] = trans_off[4]*X + trans_off[5]*Y + trans_off[6]*d + trans_off[7];
    int z_offset = ((n * top_channel + 2) * H + y) * W + x;
    transformed_points[z_offset] = trans_off[8]*X + trans_off[9]*Y + trans_off[10]*d + trans_off[11];
   }
}

template <typename Dtype>
void GeoTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int top_channel = top[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int n_threads = num * height * width;

  // bottom[0] --> DepthMap (N,1,H,W)
  // bottom[1] --> Transformation matrix (N,1,4,4)
  // bottom[2] --> camera intrinsic coefficient (N,4,1,1)
  // top[0]    --> transformed 3D points (N,3,H,W)

  GeoTransform <<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
                    n_threads, height, width, top_channel, bottom[0]->gpu_data(), bottom[1]->gpu_data(), bottom[2]->gpu_data(), 
                    top[0]->mutable_gpu_data());
}



template <typename Dtype>
__global__ void GetGradient(const int nthreads,
    const Dtype* const top_diff, const Dtype* const depths, const Dtype* const transformation, const Dtype* const camIntrinsic, 
                  const int height, const int width, const int top_channel, Dtype* const depths_diff, Dtype* const transformation_diff, Dtype* const camIntrinsic_diff,
                  const bool propagate_down_0, const bool propagate_down_1,const bool propagate_down_2
                  ){

    CUDA_KERNEL_LOOP(index, nthreads) {
    const int x = index % width;
    const int y = (index / width) % height;
    // const int top_channel = 4;
    const int n = index / width / height ;

    const Dtype* const trans_off = transformation + n*16; 
    // const Dtype* const trans_diff_off = transformation_diff + n*16; 
    const Dtype* const camIntrin_off = camIntrinsic + n*4; 
    // const Dtype* const camIntrin_diff_off = camIntrinsic_diff + n*4; 
    int depth_offset = (n*height+y)*width+x;

    const float fx = camIntrin_off[0];
    const float fy = camIntrin_off[1];
    const float cx = camIntrin_off[2];
    const float cy = camIntrin_off[3];

    // Setup base grid
    const float base_X = (x-cx)/fx;
    const float base_Y = (y-cy)/fy;
    const Dtype depth = depths[depth_offset];

    int x_offset = ((n*top_channel + 0)*height+y)*width+x;
    int y_offset = ((n*top_channel + 1)*height+y)*width+x;
    int z_offset = ((n*top_channel + 2)*height+y)*width+x;



    if (propagate_down_0){
      
      // bottom[0] dLdD
      Dtype depth_diff_val(0);
      float dX_dD = trans_off[0] * base_X + trans_off[1] * base_Y + trans_off[2];
      depth_diff_val += top_diff[x_offset] * dX_dD;

      
      float dY_dD = trans_off[4] * base_X + trans_off[5] * base_Y + trans_off[6];
      depth_diff_val += top_diff[y_offset] * dY_dD;

      
      float dZ_dD = trans_off[8] * base_X + trans_off[9] * base_Y + trans_off[10];
      depth_diff_val += top_diff[z_offset] * dZ_dD;

      depths_diff[depth_offset] = depth_diff_val;
    }


    if (propagate_down_1){
      // bottom[1] dLdT
      caffe_gpu_atomic_add( top_diff[x_offset] * base_X * depth, transformation_diff + n*16 + 0);
      caffe_gpu_atomic_add( top_diff[x_offset] * base_Y * depth, transformation_diff + n*16 + 1);
      caffe_gpu_atomic_add( top_diff[x_offset] * depth, transformation_diff + n*16 + 2);
      caffe_gpu_atomic_add( top_diff[x_offset] * Dtype(1), transformation_diff + n*16 + 3);

      caffe_gpu_atomic_add( top_diff[y_offset] * base_X * depth, transformation_diff + n*16 + 4);
      caffe_gpu_atomic_add( top_diff[y_offset] * base_Y * depth, transformation_diff + n*16 + 5);
      caffe_gpu_atomic_add( top_diff[y_offset] * depth, transformation_diff + n*16 + 6);
      caffe_gpu_atomic_add( top_diff[y_offset] * Dtype(1), transformation_diff + n*16 + 7);

      caffe_gpu_atomic_add( top_diff[z_offset] * base_X * depth, transformation_diff + n*16 + 8);
      caffe_gpu_atomic_add( top_diff[z_offset] * base_Y * depth, transformation_diff + n*16 + 9);
      caffe_gpu_atomic_add( top_diff[z_offset] * depth, transformation_diff + n*16 + 10);
      caffe_gpu_atomic_add( top_diff[z_offset] * Dtype(1), transformation_diff + n*16 + 11);
    }

    if (propagate_down_2){
      // bottom[2] dLdK
      Dtype cx_diff_val(0); 
      cx_diff_val += top_diff[x_offset] * trans_off[0];
      cx_diff_val += top_diff[y_offset] * trans_off[4];
      cx_diff_val += top_diff[z_offset] * trans_off[8]; 
      cx_diff_val *= -depth/fx;
      caffe_gpu_atomic_add( cx_diff_val, camIntrinsic_diff + n*4 + 2); 

      Dtype cy_diff_val(0); 
      cy_diff_val += top_diff[x_offset] * trans_off[1]; 
      cy_diff_val += top_diff[y_offset] * trans_off[5]; 
      cy_diff_val += top_diff[z_offset] * trans_off[9]; 
      cy_diff_val *= -depth/fy;
      caffe_gpu_atomic_add( cy_diff_val, camIntrinsic_diff + n*4 + 3); 

      Dtype fx_diff_val(0); 
      fx_diff_val += top_diff[x_offset] * trans_off[0]; 
      fx_diff_val += top_diff[y_offset] * trans_off[4]; 
      fx_diff_val += top_diff[z_offset] * trans_off[8]; 
      fx_diff_val *= - base_X / fx * depth;
      caffe_gpu_atomic_add( fx_diff_val, camIntrinsic_diff + n*4 + 0);

      Dtype fy_diff_val(0); 
      fy_diff_val += top_diff[x_offset] * trans_off[1]; 
      fy_diff_val += top_diff[y_offset] * trans_off[5]; 
      fy_diff_val += top_diff[z_offset] * trans_off[9]; 
      fy_diff_val *= - base_Y / fy * depth;
      caffe_gpu_atomic_add( fy_diff_val, camIntrinsic_diff + n*4 + 1);
    }

  }
}

template <typename Dtype>
void GeoTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{  
  const Dtype* top_diff = top[0]->gpu_diff();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int top_channel = top[0]->channels();
  int n_threads = num * height * width;

  const Dtype* depths = bottom[0]->gpu_data();
  const Dtype* transformation = bottom[1]->gpu_data();
  const Dtype* camIntrinsic = bottom[2]->gpu_data();

  Dtype* depths_diff = bottom[0]->mutable_gpu_diff();
  Dtype* transformation_diff = bottom[1]->mutable_gpu_diff();
  Dtype* camIntrinsic_diff = bottom[2]->mutable_gpu_diff();

  caffe_gpu_set(bottom[0]->count(), (Dtype)0., depths_diff);
  caffe_gpu_set(bottom[1]->count(), (Dtype)0., transformation_diff);
  caffe_gpu_set(bottom[2]->count(), (Dtype)0., camIntrinsic_diff);

  GetGradient<Dtype><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
                        n_threads, top_diff, depths, transformation, camIntrinsic, 
                  height, width, top_channel, depths_diff, transformation_diff, camIntrinsic_diff, 
                  propagate_down[0], propagate_down[1], propagate_down[2]);
  CUDA_POST_KERNEL_CHECK; 
}

INSTANTIATE_LAYER_GPU_FUNCS(GeoTransformLayer);

}  // namespace caffe
