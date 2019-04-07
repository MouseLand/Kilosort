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
const int  Nthreads = 1024, NrankMax = 3;
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	Conv1D(const double *Params, const float *data, const float *W, float *conv_sig){    
  volatile __shared__ float  sW[81*NrankMax], sdata[(Nthreads+81)*NrankMax]; 
  float x, y;
  int tid, tid0, bid, i, nid, Nrank, NT, nt0;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  NT      	=   (int) Params[0];
  nt0       = (int) Params[2];
  Nrank     = (int) Params[3];  
   
  if(tid<nt0*Nrank)
      sW[tid]= W[tid%nt0 + (tid/nt0) * nt0];
  __syncthreads();
  
  tid0 = 0;
  while (tid0<NT-Nthreads-nt0+1){
	  if (tid<nt0*NrankMax) 
          sdata[tid%nt0 + (tid/nt0)*(Nthreads+nt0)] = 
			data[tid0 + tid%nt0+ NT*bid];
	  
      #pragma unroll 3
      for(nid=0;nid<Nrank;nid++){
          sdata[tid + nt0+nid*(Nthreads+nt0)] = data[nt0+tid0 + tid+ NT*bid];
	  }
	  __syncthreads();
      
	  x = 0.0f;
      for(nid=0;nid<Nrank;nid++){
          y = 0.0f;
		  #pragma unroll 4
          for(i=0;i<nt0;i++)
              y    += sW[i + nid*nt0] * sdata[i+tid + nid*(Nthreads+nt0)];

           x += y*y;
      }
      conv_sig[tid0  + tid + NT*bid]   = x;
      
      tid0+=Nthreads;
      __syncthreads();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  /* Initialize the MathWorks GPU API. */
  mxInitGPU();

  /* Declare input variables*/
  double *Params, *d_Params;
  unsigned int NT, Nchan;

  
  /* read Params and copy to GPU */
  // NT, Nchan, nt0, Nrank
  
  Params  	= (double*) mxGetData(prhs[0]);
  NT		= (unsigned int) Params[0];
  Nchan     = (unsigned int) Params[1];
        
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);

   /* collect input GPU variables*/
  mxGPUArray const  *W,  *data;
  const float     *d_W, *d_data;
  mxGPUArray *dout;
  
  data       = mxGPUCreateFromMxArray(prhs[1]);
  d_data     = (float const *)(mxGPUGetDataReadOnly(data));
  W             = mxGPUCreateFromMxArray(prhs[2]);
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  
  /* allocate new GPU variables*/  
  const mwSize dimsD[] 	= {NT , Nchan};
  float *d_dout;  
  dout  = mxGPUCreateGPUArray(2,  dimsD, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  d_dout     = (float *)(mxGPUGetData(dout));
  
  
  // filter the data with the temporal templates
  Conv1D<<<Nchan, Nthreads>>>(d_Params, d_data, d_W, d_dout);
  
  // dWU stays a GPU array
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(dout);
  
  cudaFree(d_Params);
  
  mxGPUDestroyGPUArray(dout);  
  mxGPUDestroyGPUArray(W);
  mxGPUDestroyGPUArray(data);

}