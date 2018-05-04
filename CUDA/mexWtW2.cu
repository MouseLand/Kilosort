/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
using namespace std;

const int Nthreads = 1024,   nblock = 32;
//////////////////////////////////////////////////////////////////////////////////////////

__global__ void	crossFilter(const double *Params, const float *W1, const float *W2,
        const float *UtU, float *WtW){    
  __shared__ float shW1[nblock*81], shW2[nblock*81]; 

  float x;
  int nt0, tidx, tidy , bidx, bidy, i, NT, Nfilt, t;

  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;
  bidx 		= blockIdx.x;
  bidy 		= blockIdx.y;
  
  Nfilt = (int) Params[1];
  nt0       = (int) Params[9];
  
  while(tidx<nt0){
    shW1[tidx + tidy * nt0] = W1[tidx + (tidy+bidx*nblock) * nt0];
    shW2[tidx + tidy * nt0] = W2[tidx + (tidy+bidy*nblock) * nt0];
    tidx+= nblock;
  }
  tidx 		= threadIdx.x;
  __syncthreads();
	 	 
  for(i=0;i<2*nt0-1;i++){
      x = 0.0f;
      if(i<nt0)
          for(t=0;t<i+1;t++)
              x += shW1[t + nt0 * tidx] * shW2[t + (nt0-i-1) + nt0 * tidy];
      else
          for(t=i-nt0+1;t<nt0;t++)
              x += shW1[t + nt0 * tidx] * shW2[t + (nt0-i-1) + nt0 * tidy];
      WtW[tidx+bidx*nblock + (tidy + bidy*nblock)*Nfilt +  i*Nfilt*Nfilt] =
              x * UtU[tidx+bidx*nblock + (tidy + bidy*nblock)*Nfilt];
  }
}


//////////////////////////////////////////////////////////////////////////////////////////

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare input variables*/
  double *Params, *d_Params;
  int nt0, Nfilt, NT;

  /* Initialize the MathWorks GPU API. */
  mxInitGPU();

  /* read Params and copy to GPU */
  Params  	= (double*) mxGetData(prhs[0]);
  NT		= (int) Params[0];
  Nfilt		= (int) Params[1];
  nt0       = (int) Params[9];
  
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);

  /* collect input GPU variables*/
  mxGPUArray const  *W1, *W2,   *UtU;
  const float     *d_W1,*d_W2, *d_UtU;
  
  W1             = mxGPUCreateFromMxArray(prhs[1]);
  d_W1        	= (float const *)(mxGPUGetDataReadOnly(W1));
  W2             = mxGPUCreateFromMxArray(prhs[2]);
  d_W2        	= (float const *)(mxGPUGetDataReadOnly(W2));
  UtU       	= mxGPUCreateFromMxArray(prhs[3]);
  d_UtU     	= (float const *)(mxGPUGetDataReadOnly(UtU));


  mxGPUArray *WtW;
  float  *d_WtW;
  const mwSize dimsu[] 	= {Nfilt, Nfilt, 2*nt0-1}; 
  WtW 		= mxGPUCreateGPUArray(3, dimsu, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
  d_WtW 		= (float *)(mxGPUGetData(WtW));

  dim3 grid(Nfilt/nblock, Nfilt/nblock);
  dim3 block(nblock, nblock);
  crossFilter<<<grid, block>>>(d_Params, d_W1, d_W2, d_UtU, d_WtW); 

  plhs[0] 	= mxGPUCreateMxArrayOnGPU(WtW);

  cudaFree(d_Params);
  mxGPUDestroyGPUArray(WtW);
  mxGPUDestroyGPUArray(W1);
  mxGPUDestroyGPUArray(W2);
  mxGPUDestroyGPUArray(UtU);
  
}
