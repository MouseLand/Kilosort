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

const int  Nthreads = 1024;

__global__ void	recWU(const double *Params,  const float *W, const float *U,
        const float *mu, float *dWU){
    
    int nt0, tidx, bid, tidy, k, Nchan, Nfilt, Nrank;
    float X;
    
    nt0       = (int) Params[4];
    Nchan     = (int) Params[9];
    Nfilt     =   (int) Params[1];
    Nrank     = (int) Params[6];
    
    tidx 	  = threadIdx.x;
    tidy      = threadIdx.y;
    bid      = blockIdx.x;
    
    while (tidy<Nchan){
        X = 0.0f;
        for (k=0;k<Nrank;k++)
            X += W[tidx + k *Nfilt* nt0 + nt0*bid] *
                    U[tidy + k * Nfilt * Nchan + Nchan*bid];
        
        dWU[tidx + nt0 * tidy + bid*Nchan*nt0] = X * mu[bid];
        tidy += blockDim.y;
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
  int nt0, Nfilt, Nchan;

  /* read Params and copy to GPU */
  Params  	= (double*) mxGetData(prhs[0]);
  Nfilt     = (int) Params[1];
  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];

  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);

   /* collect input GPU variables*/
  mxGPUArray const *W, *U, *mu;
  mxGPUArray  *dWU;
  float *d_dWU;
  const float *d_W, *d_U, *d_mu;  

  W       = mxGPUCreateFromMxArray(prhs[1]);
  d_W     = (float const *)(mxGPUGetDataReadOnly(W));
  U       = mxGPUCopyFromMxArray(prhs[2]);
  d_U     = (float const *)(mxGPUGetDataReadOnly(U));  
  mu       = mxGPUCopyFromMxArray(prhs[3]);
  d_mu     = (float const *)(mxGPUGetDataReadOnly(mu));
  
  const mwSize dimsdWU[] 	= {nt0, Nchan, Nfilt}; 
  
  dWU  = mxGPUCreateGPUArray(3,  dimsdWU, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
  d_dWU 		= (float *)(mxGPUGetData(dWU));  
  cudaMemset(d_dWU,    0, nt0*Nchan * Nfilt* sizeof(float));
  
  dim3 tpS(nt0, Nthreads/nt0);
  
  recWU<<<Nfilt, tpS>>>(d_Params, d_W, d_U, d_mu, d_dWU);
  
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(dWU);
  
  cudaFree(d_Params);  
  
  mxGPUDestroyGPUArray(dWU);
  mxGPUDestroyGPUArray(W);
  mxGPUDestroyGPUArray(U);
  mxGPUDestroyGPUArray(mu);
  
}