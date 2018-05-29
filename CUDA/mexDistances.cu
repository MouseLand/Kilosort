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


//////////////////////////////////////////////////////////////////////////////////////////
__global__ void computeCost(const double *Params, const float *Ws, const float *mus, 
        const float *W, const float *mu, const int *ioff, const bool *iW, float *cmax){
    
  int tid, bid, Nspikes, Nfeatures, NfeatW, Nthreads, k;
  float xsum = 0.0f, Ci; 
  
  Nspikes               = (int) Params[0];
  Nfeatures             = (int) Params[1];
  NfeatW                = (int) Params[4];
  Nthreads              = blockDim.x;  
    
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  while(tid<Nspikes){
      if (iW[tid + bid*Nspikes]){
          xsum = 0.0f;
          for (k=0;k<Nfeatures;k++)
              xsum += Ws[k + Nfeatures * tid] * W[k + ioff[tid] +  NfeatW * bid];
          Ci = mu[bid]*mu[bid] + mus[tid]*mus[tid] -2*mus[tid]*mu[bid]*xsum;
          cmax[tid + bid*Nspikes] = Ci;          
      }
      tid+= Nthreads;
  }  
}


//////////////////////////////////////////////////////////////////////////////////////////
__global__ void bestFilter(const double *Params, const bool *iW, const float *cmax, const float *mus,
        int *id, float *x){
    
  int tid,tind,bid, ind, Nspikes, Nfilters, Nthreads, Nblocks;
  float max_running = 0.0f; 
  
  Nspikes               = (int) Params[0];
  Nfilters              = (int) Params[2];
  Nthreads              = blockDim.x;
  Nblocks               = gridDim.x;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  tind = tid + bid * Nthreads;
  
  while (tind<Nspikes){
      max_running = mus[tind] * mus[tind];
      id[tind] = 0;
      
      for(ind=0; ind<Nfilters; ind++)
          if (iW[tind + ind*Nspikes])
              if (cmax[tind + ind*Nspikes] < max_running){
                  id[tind] = ind;
                  max_running = cmax[tind + ind*Nspikes];
              }
      x[tind] = max_running;
        
      tind += Nblocks*Nthreads; 
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
  int  Nspikes, Nfilters;
  
  /* Initialize the MathWorks GPU API. */
  mxInitGPU();

  /* read Params and copy to GPU */
  Params                = (double*) mxGetData(prhs[0]);
  Nspikes               = (int) Params[0];
  Nfilters              = (int) Params[2];
  
  // copy Params to GPU
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);
  
  /* collect input GPU variables*/
  mxGPUArray const  *W, *Ws, *ioff, *iW, *mu, *mus;
  const float *d_W, *d_Ws, *d_mu, *d_mus;
  const int *d_ioff;
  const bool *d_iW;
    
  // these come as const GPU Arrays, just transfer them over
  Ws            = mxGPUCreateFromMxArray(prhs[1]);
  W             = mxGPUCreateFromMxArray(prhs[2]);  
  ioff          = mxGPUCreateFromMxArray(prhs[3]);  
  iW            = mxGPUCreateFromMxArray(prhs[4]);
  mus            = mxGPUCreateFromMxArray(prhs[5]);
  mu            = mxGPUCreateFromMxArray(prhs[6]);

  d_Ws      = (float const *)(mxGPUGetDataReadOnly(Ws));
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  d_ioff        = (int const *)  (mxGPUGetDataReadOnly(ioff));
    // this has a one for filter - spike combinations to be considered
  d_iW          = (bool const *)  (mxGPUGetDataReadOnly(iW));
  d_mu          = (float const *)  (mxGPUGetDataReadOnly(mu));
  d_mus          = (float const *)  (mxGPUGetDataReadOnly(mus));
  
  /* Define new GPU variables*/
  float *d_cmax, *d_x;
  int *d_id;
  
  // allocate a lot of GPU variables
  cudaMalloc(&d_cmax,    Nspikes * Nfilters *  sizeof(float));
  cudaMalloc(&d_id,      Nspikes  *  sizeof(int));  
  cudaMalloc(&d_x,      Nspikes  * sizeof(float));
   
  // get list of cmaxes for each combination of neuron and filter
  computeCost<<<Nfilters, 1024>>>(d_Params, d_Ws, d_mus, d_W, d_mu, d_ioff, 
          d_iW, d_cmax);

  // loop through cmax to find best template
  bestFilter<<<40, 256>>>(d_Params, d_iW, d_cmax, d_mus, d_id, d_x);  
  
  // put these ones on the CPU side: id, cmax, cf, nsp 
  int *id;
  float *x;
    const mwSize dimst[] 	= {Nspikes,1};  

  plhs[0]   = mxCreateNumericArray(2, dimst,  mxINT32_CLASS,  mxREAL);
  plhs[1]   = mxCreateNumericArray(2, dimst, mxSINGLE_CLASS, mxREAL);    

  id     = (int*) mxGetData(plhs[0]);  
  x      = (float*) mxGetData(plhs[1]);  
  
  
  cudaMemcpy(id,   d_id,  Nspikes * sizeof(int),   cudaMemcpyDeviceToHost);
  cudaMemcpy(x, d_x,Nspikes * sizeof(float),  cudaMemcpyDeviceToHost);  
  
  //we are done, clear everything from the GPU
  cudaFree(d_Params);
  cudaFree(d_cmax);
  cudaFree(d_id);
  cudaFree(d_x);

  //do this for the constant variables
  mxGPUDestroyGPUArray(W);    
  mxGPUDestroyGPUArray(Ws);    
  mxGPUDestroyGPUArray(ioff);  
  mxGPUDestroyGPUArray(iW);  
  mxGPUDestroyGPUArray(mu);  
  mxGPUDestroyGPUArray(mus);  

  
}
