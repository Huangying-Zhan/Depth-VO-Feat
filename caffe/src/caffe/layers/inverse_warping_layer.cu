#include <cfloat>
#include <vector>

#include "caffe/layers/inverse_warping_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void InverseWarping(const int nthreads,
  const int top_channel, const int height, const int width,
    const Dtype* const U, const Dtype* const proj_xy, Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
    const int x = index % width;
    const int y = (index / width) % height;
    const int n = index / width / height ;

    const int proj_x_offset = ((n * 2 + 0) * height + y) * width + x;
    const int proj_y_offset = ((n * 2 + 1) * height + y) * width + x;
    
    float xx = proj_xy[proj_x_offset];
    float yy = proj_xy[proj_y_offset];
    
    int x1 = floorf(xx);
    int x2 = x1+1;

    int y1 = floorf(yy);
    int y2 = y1+1;

    float wx2 = xx - float(x1);
    float wx1 = float(x2)-xx;
    float wy2 = yy - float(y1);
    float wy1 = float(y2)-yy;
    
  
    
    for (int cc = 0; cc<top_channel; cc++){
    int off =  (n * top_channel +  cc)*height*width;
                top_data[x + y * width +off] = 0;
             
        if(x1 >= 0 && x1 <= width-1 && y1 >= 0 && y1 <= height-1  )  
      top_data[x + y * width +off] += wx1 * wy1 * U[x1 + y1 *width + off] ;
        if(x1 >= 0 && x1 <= width-1 && y2 >= 0 && y2 <= height-1 ) 
      top_data[x + y * width +off] += wx1 * wy2 * U[x1 + y2 *width + off] ;
        if(x2 >= 0 && x2 <= width-1 && y1 >= 0 && y1 <= height-1  ) 
      top_data[x + y * width +off] += wx2 * wy1 * U[x2 + y1 *width + off] ;
        if(x2 >= 0 && x2 <= width-1 && y2 >= 0 && y2 <= height-1  )  
      top_data[x + y * width +off] += wx2 * wy2 * U[x2 + y2 *width + off] ;
     }
  }
}



template <typename Dtype>
void InverseWarpingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();

  int top_channel = top[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int n_threads = num * height * width;

  // bottom[0] --> Image (N,C,H,W)
  // bottom[1] --> projection coordinate (N,K,H,W)
  // top[0]    --> Warp iamge (N,C,H,W)


  InverseWarping <<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
                    n_threads, top_channel, height, width, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
                    top[0]->mutable_gpu_data());



}





template <typename Dtype>
__global__ void GetGradient(
      const int nthreads, const int top_channel, const int height, const int width, 
      const Dtype* const U, const Dtype* const proj_xy,
      const Dtype* const top_diff, 
      Dtype* const U_diff, Dtype* const proj_xy_diff,
      const bool propagate_down_0, const bool propagate_down_1) {
    CUDA_KERNEL_LOOP(index, nthreads) {

      const int x = index % width;
      const int y = (index / width) % height;
      const int n = index / width / height ;

      const int proj_x_offset = ((n * 2 + 0) * height + y) * width + x;
      const int proj_y_offset = ((n * 2 + 1) * height + y) * width + x;
      
      float xx = proj_xy[proj_x_offset];
      float yy = proj_xy[proj_y_offset];
      
      int x1 = floorf(xx);
      int x2 = x1+1;
      int y1 = floorf(yy);
      int y2 = y1+1;

      float wx2 = xx - float(x1);
      float wx1 = float(x2)-xx;
      float wy2 = yy - float(y1);
      float wy1 = float(y2)-yy;

      Dtype topLeftProduct(0); 
      Dtype topRightProduct(0); 
      Dtype bottomLeftProduct(0); 
      Dtype bottomRightProduct(0); 
      


      
        // if (index==0){printf("Propagate_down_0: %d\n", propagate_down_0);}
        

        for (int cc = 0; cc<top_channel; cc++){
        int off =  (n*top_channel +  cc)*height*width;
        Dtype top_diff_this = top_diff[off + width * y + x];
            if(x1 >= 0 && x1 <= width-1 && y1 >= 0 && y1 <= height-1  ) {
              // dLdU
              if (propagate_down_0){ caffe_gpu_atomic_add(top_diff_this*wx1*wy1, U_diff + x1 + y1 * width + off ); }
              // for dLd[x,y] future use
              topLeftProduct += top_diff_this * U[off + width * y1 + x1];

            }
            if(x1 >= 0 && x1 <= width-1 && y2 >= 0 && y2 <= height-1 ) {
              // dLdU
              if (propagate_down_0){ caffe_gpu_atomic_add(top_diff_this*wx1*wy2, U_diff +x1 + y2 * width + off );}
              // for dLd[x,y] future use
              bottomLeftProduct += top_diff_this * U[off + width * y2 + x1];

            }
          
            if(x2 >= 0 && x2 <= width-1 && y1 >= 0 && y1 <= height-1  ) {
              // dLdU
              if (propagate_down_0){ caffe_gpu_atomic_add(top_diff_this*wx2*wy1, U_diff +x2 + y1 * width + off );}
              // for dLd[x,y] future use
              topRightProduct += top_diff_this * U[off + width * y1 + x2];

            }
          
            if(x2 >= 0 && x2 <= width-1 && y2 >= 0 && y2 <= height-1  )  {
              // dLdU
              if (propagate_down_0){ caffe_gpu_atomic_add(top_diff_this*wx2*wy2, U_diff +x2 + y2 * width + off );}
              // for dLd[x,y] future use
              bottomRightProduct += top_diff_this * U[off + width * y2 + x2];
            }
        }
      
      
      // dLd[x,y]
      if (propagate_down_1){
        // if (index==0){printf("Propagate_down_1: %d\n", propagate_down_1);}
        proj_xy_diff[proj_x_offset] = (topRightProduct - topLeftProduct) * wy1 + (bottomRightProduct - bottomLeftProduct) * wy2;
        proj_xy_diff[proj_y_offset] = (bottomLeftProduct - topLeftProduct) * wx1 + (bottomRightProduct - topRightProduct) * wx2;  
        // if (index==0){printf("grad : %f\n", proj_xy_diff[proj_y_offset]);}
      }
      
    }
    
}





template <typename Dtype>
void InverseWarpingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  const Dtype* top_diff = top[0]->gpu_diff();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int top_channel = top[0]->channels();
  int n_threads = num * height * width;

  const Dtype* U = bottom[0]->gpu_data();
  const Dtype* proj_xy = bottom[1]->gpu_data();

  bool propagate_down_0 = propagate_down[0];
  bool propagate_down_1 = propagate_down[1];

  Dtype* U_diff = bottom[0]->mutable_gpu_diff();
  Dtype* proj_xy_diff = bottom[1]->mutable_gpu_diff();

  caffe_gpu_set(bottom[0]->count(), (Dtype)0., U_diff);
  caffe_gpu_set(bottom[1]->count(), (Dtype)0., proj_xy_diff);

  GetGradient<Dtype><<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
                        n_threads, top_channel, height, width,
                        U, proj_xy,  
                        top_diff, U_diff, proj_xy_diff,
                        propagate_down_0, propagate_down_1);

  CUDA_POST_KERNEL_CHECK; 



}


INSTANTIATE_LAYER_GPU_FUNCS(InverseWarpingLayer);
}

