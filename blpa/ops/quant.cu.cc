// --Ayan Chakrabarti <ayan@wustl.edu>
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

typedef float float32;
typedef unsigned char uint8;


__global__ void save_kern4(uint8* lhs,
			   const float32 * rhs, const float32 * bias,
			   int numEl, int numCh) {

  int i, ch; float32 f, shift; uint8 out;
  
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < numEl/2; j += blockDim.x * gridDim.x) {

    i = j*2; ch = i%numCh;
    f = floorf((rhs[i]-1.0e-6) * (16.0/6.0));
    shift = floorf(bias[ch]*(16.0/6.0));
    shift = fmaxf(1.0,8.0-shift);

    f += shift; f = fmaxf(0.0,f); f = fminf(15.0,f);
    out = ((uint8) f) << 4;

    i++; ch++;
    f = floorf((rhs[i]-1.0e-6) * (16.0/6.0));
    shift = floorf(bias[ch]*(16.0/6.0));
    shift = fmaxf(1.0,8.0-shift);

    f += shift; f = fmaxf(0.0,f); f = fminf(15.0,f);
    out |= (uint8) f;

    lhs[j] = out;    
  }
  
}

__global__ void rest_kern4(float32 * act, float32 * Rm,
			   const uint8 * var, const float32 * bias,
			   int numEl, int numCh) {

  int i, ch; float32 f, shift; uint8 v;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < numEl/2; j += blockDim.x * gridDim.x) {

    i = j*2; ch = i%numCh;
    shift = floorf(bias[ch]*(16.0/6.0));
    shift = fmaxf(1.0,8.0-shift);
    v = var[j]>>4;
    Rm[i] = (v >= ((uint8) shift)) ? 1.0 : 0.0;
    f = (((float32)v) - shift + 0.5)*(6.0/16.0);
    act[i] = f;

    i++; ch++;
    shift = floorf(bias[ch]*(16.0/6.0));
    shift = fmaxf(1.0,8.0-shift);
    v = var[j] & 0xf;
    Rm[i] = (v >= ((uint8) shift)) ? 1.0 : 0.0;
    f = (((float32)v) - shift + 0.5)*(6.0/16.0);
    act[i] = f;
  }
  
}


__global__ void save_kern8(uint8* lhs,
			   const float32 * rhs, const float32 * bias,
			   int numEl, int numCh) {

  int ch; float32 f, shift;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEl; i += blockDim.x * gridDim.x) {
    ch = i%numCh;
    f = floorf((rhs[i]-1.0e-6) * (256.0/6.0));
    shift = floorf(bias[ch]*(256.0/6.0));
    shift = fmaxf(1.0,128.0-shift);

    f += shift; f = fmaxf(0.0,f); f = fminf(255.0,f);
    lhs[i] = (uint8) f;
  }
  
}

__global__ void rest_kern8(float32 * act, float32 * Rm,
			   const uint8 * var, const float32 * bias,
			   int numEl, int numCh) {

  int ch; float32 f, shift; uint8 v;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEl; i += blockDim.x * gridDim.x) {
    ch = i%numCh;
    shift = floorf(bias[ch]*(256.0/6.0));
    shift = fmaxf(1.0,128.0-shift);
    v = var[i];

    Rm[i] = (v >= ((uint8) shift)) ? 1.0 : 0.0;
    f = (((float32)v) - shift + 0.5)*(6.0/256.0);
    act[i] = f;
  }
  
}



void gpuSave(const GPUDevice& d, uint8* lhs,
	     const float32 * rhs, const float32 * bias,
	     int numEl, int numCh, int nBits) {

  if(nBits == 8) {
    CudaLaunchConfig config = GetCudaLaunchConfig(numEl, d);
    save_kern8<<<config.block_count, config.thread_per_block, 0, d.stream()>>>
      (lhs,rhs,bias,numEl,numCh);
  } else if(nBits == 4) {
    CudaLaunchConfig config = GetCudaLaunchConfig(numEl/2, d);
    save_kern4<<<config.block_count, config.thread_per_block, 0, d.stream()>>>
      (lhs,rhs,bias,numEl,numCh);
  }


}

void gpuRest(const GPUDevice& d, float32 * act, float32 * Rm,
	     const uint8 * var, const float32 * bias,
	     int numEl, int numCh, int nBits) {

  if(nBits == 8) {
    CudaLaunchConfig config = GetCudaLaunchConfig(numEl, d);
    rest_kern8<<<config.block_count, config.thread_per_block, 0, d.stream()>>>
      (act,Rm,var,bias,numEl,numCh);
  } else if(nBits == 4) {
    CudaLaunchConfig config = GetCudaLaunchConfig(numEl/2, d);
    rest_kern4<<<config.block_count, config.thread_per_block, 0, d.stream()>>>
      (act,Rm,var,bias,numEl,numCh);
  }
}

#endif


