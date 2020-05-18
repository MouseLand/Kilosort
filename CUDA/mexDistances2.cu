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
        const float *W, const float *mu, const bool *iMatch, 
        const int *iC, const int *Wh, float *cmax){
    
  int j, tid, bid, Nspikes, my_chan, this_chan, Nchan, NrankPC, NchanNear, Nthreads, k;
  float xsum = 0.0f, Ci; 
  
  Nspikes               = (int) Params[0];  //more accurately, number of comparisons, Nfilt*Nbatch
  Nchan                 = (int) Params[7];
  NrankPC                 = (int) Params[1];
  NchanNear                 = (int) Params[6];
  Nthreads              = blockDim.x;  
  
    
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  while(tid<Nspikes){
      my_chan = Wh[tid];      
      if (iMatch[my_chan + bid*Nchan]){
          xsum = 0.0f;
          for (k=0;k<NchanNear;k++){
              this_chan = iC[k + NchanNear * my_chan];
              for (j=0;j<NrankPC;j++)
                    xsum += Ws[j + NrankPC*k + NrankPC*NchanNear * tid] *
                            W[j + NrankPC*this_chan + NrankPC*Nchan * bid];
              
          }
          
          Ci = mu[bid]*mu[bid] + mus[tid]*mus[tid] -2*mus[tid]*mu[bid]*xsum;
          cmax[tid + bid*Nspikes] = Ci;          
      }
      tid+= Nthreads;
  }  
}


//////////////////////////////////////////////////////////////////////////////////////////
__global__ void bestFilter(const double *Params, const bool *iMatch,
        const int *Wh, const float *cmax, const float *mus, int *id, float *x){
    
  int tid,tind,bid, my_chan, ind, Nspikes, Nfilters, Nthreads, Nchan, Nblocks;
  float max_running = 0.0f; 
  
  Nspikes               = (int) Params[0];
  Nfilters              = (int) Params[2];
  Nchan                 = (int) Params[7];
  Nthreads              = blockDim.x;
  Nblocks               = gridDim.x;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  tind = tid + bid * Nthreads;
  
  while (tind<Nspikes){      
      max_running = mus[tind] * mus[tind];
      id[tind] = 0;
      my_chan = Wh[tind];      
      for(ind=0; ind<Nfilters; ind++)          
          if (iMatch[my_chan + ind * Nchan])
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
  unsigned int  Nspikes, Nfilters;
  
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
  mxGPUArray const  *W, *Ws, *Wh, *mu, *mus, *iC, *iMatch;
  const float *d_W, *d_Ws, *d_mu, *d_mus;
  const int *d_iC, *d_Wh;
  const bool *d_iMatch;
    
  // these come as const GPU Arrays, just transfer them over
  Ws            = mxGPUCreateFromMxArray(prhs[1]);
  W             = mxGPUCreateFromMxArray(prhs[2]);  
  iMatch        = mxGPUCreateFromMxArray(prhs[3]);  
  iC            = mxGPUCreateFromMxArray(prhs[4]);
  Wh            = mxGPUCreateFromMxArray(prhs[5]);
  mus           = mxGPUCreateFromMxArray(prhs[6]);
  mu            = mxGPUCreateFromMxArray(prhs[7]);

  d_Ws          = (float const *)(mxGPUGetDataReadOnly(Ws));
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  d_iMatch      = (bool const *)(mxGPUGetDataReadOnly(iMatch));
  d_iC          = (int const *)(mxGPUGetDataReadOnly(iC));
  d_Wh          = (int const *)(mxGPUGetDataReadOnly(Wh));
  d_mu          = (float const *)(mxGPUGetDataReadOnly(mu));
  d_mus         = (float const *)(mxGPUGetDataReadOnly(mus));
  
  /* Define new GPU variables*/
  float *d_cmax, *d_x;
  int *d_id;
  
  // allocate a lot of GPU variables
  cudaMalloc(&d_cmax,    Nspikes * Nfilters *  sizeof(float));
  cudaMalloc(&d_id,      Nspikes  *  sizeof(int));  
  cudaMalloc(&d_x,      Nspikes  * sizeof(float));
   
  // get list of cmaxes for each combination of neuron and filter
  computeCost<<<Nfilters, 1024>>>(d_Params, d_Ws, d_mus, d_W, d_mu, 
          d_iMatch, d_iC, d_Wh, d_cmax);

  // loop through cmax to find best template
  bestFilter<<<40, 256>>>(d_Params, d_iMatch, d_Wh, d_cmax, d_mus, d_id, d_x);  
  
  // put these ones on the CPU side: id, cmax, cf, nsp 
  int *id;
  float *x;
  const mwSize dimst[] 	= {Nspikes,1};
  plhs[0]   = mxCreateNumericArray(2, dimst,  mxINT32_CLASS, mxREAL);
  plhs[1]   = mxCreateNumericArray(2, dimst, mxSINGLE_CLASS, mxREAL);    

  id     = (int*) mxGetData(plhs[0]);  
  x      = (float*) mxGetData(plhs[1]);  
  
  
  cudaMemcpy(id,   d_id,  Nspikes * sizeof(int),   cudaMemcpyDeviceToHost);
  cudaMemcpy(x,    d_x,   Nspikes * sizeof(float), cudaMemcpyDeviceToHost);  
  
  //we are done, clear everything from the GPU
  cudaFree(d_Params);
  cudaFree(d_cmax);
  cudaFree(d_id);
  cudaFree(d_x);

  //do this for the constant variables
  mxGPUDestroyGPUArray(W);    
  mxGPUDestroyGPUArray(Ws);    
  mxGPUDestroyGPUArray(Wh);  
  mxGPUDestroyGPUArray(iC);  
  mxGPUDestroyGPUArray(iMatch);  
  mxGPUDestroyGPUArray(mu);  
  mxGPUDestroyGPUArray(mus);  

  
}
