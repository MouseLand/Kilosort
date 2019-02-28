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

const int  Nthreads = 1024, maxFR = 10000, NrankMax = 6, nt0max=81, NchanMax = 32;

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  computeProjections(const double *Params, const float *dataraw, 
	const int *iC, const int *st, const int *id, const float *W, float *feat){
    
    float x;
    int tidx, tidy, tid, bid, nt0, NchanNear, j, t, NT, NrankPC;
    volatile __shared__ float sW[nt0max*NrankMax], sD[nt0max*NchanMax];
    
    NT      	=   (int) Params[0];    
    nt0       = (int) Params[4];
    NchanNear = (int) Params[10];
    NrankPC     = (int) Params[14];
    
    tidx = threadIdx.x;
    tidy = threadIdx.y;
    bid = blockIdx.x;
    
    // move wPCA to shared memory
    while (tidx<nt0){
        sW[tidx + tidy*nt0] = W[tidx + tidy*nt0];
        tidx+=blockDim.x;
    }
    tidx = threadIdx.x;    
    
    tid = tidx + tidy*blockDim.x;
    // move raw data to shared memory
    if (tid<nt0)
        for (j=0;j<NchanNear;j++)
            sD[tid + nt0*j] = dataraw[tid + st[bid] + NT * iC[j + NchanNear*id[bid]]];
    __syncthreads();
    
    x = 0.0f;
    for (t=0;t<nt0;t++)
        x += sD[t + nt0*tidx] * sW[t + nt0*tidy];
                
    feat[tidy + tidx*NrankPC + NrankPC*NchanNear*bid] = x;
    
    }

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  maxChannels(const double *Params, const float *dataraw, const float *data,
	const int *iC, int *st, int *id, int *counter){
    
  int indx, tid, tid0, nt0, i, bid, NT, Nchan, NchanNear,j,iChan, Nsum;
  double Cf, d;
  bool flag;
 
  NchanNear = (int) Params[10];    
  NT 		= (int) Params[0];
  Nchan     = (int) Params[9];
  Nsum      = (int) Params[13];  
  nt0       = (int) Params[4];
  nt0min    = (int) Params[4];
   
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  tid0 = tid + bid * blockDim.x;
  while (tid0<NT){
      for (i=0; i<Nchan;i++){          
          iChan = iC[NchanNear * i];
          Cf    = (double) data[tid0 + NT * iChan];
          flag = true;
          
          for(j=1; j<Nsum; j++){
              iChan = iC[j+ NchanNear * i];              
              if (Cf-data[tid0 + NT * iChan] > 0){
                Cf    = (double) data[tid0 + NT * iChan];
                flag = false;
                break;
              }                
          }
          
          if (flag){
              iChan = iC[NchanNear * i];
              d = (double) dataraw[tid0+nt0min + NT*iChan] - Cf;
              if (d<1e-6){
                  // this is a hit, atomicAdd and return spikes
                  indx = atomicAdd(&counter[0], 1);
                  if (indx<maxFR){
                      st[indx] = tid0;
                      id[indx] = iChan;                      
                  }                  
              }
          }          
      }
      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	max1D(const double *Params, const float *data, float *conv_sig){
    
    volatile __shared__ float  sdata[(Nthreads+81)];
    float y;
    int tid, tid0, bid, i, NT, nt0;

    tid 		= threadIdx.x;
    bid 		= blockIdx.x;
    NT      	=   (int) Params[0];
    nt0       = (int) Params[4];
    
    tid0 = 0;
    while (tid0<NT-Nthreads-nt0+1){
        if (tid<nt0)
            sdata[tid]   = data[tid0 + tid + NT*bid];        
        sdata[tid + nt0] = data[nt0+tid0 + tid+ NT*bid];
        __syncthreads();
        
        y = 0.0f;
        #pragma unroll 4
        for(i=0;i<nt0;i++)
            y    = min(y, sdata[i+tid]);
        
        conv_sig[tid0  + tid + NT*bid]   = y;

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
  int nt0, NT, Nchan, NchanNear, NrankPC;
  
  /* read Params and copy to GPU */
  Params  	= (double*) mxGetData(prhs[0]);
  NT		= (int) Params[0];
  Nchan     = (int) Params[9];
  nt0       = (int) Params[4];
  NchanNear  = (int) Params[5];
  NrankPC     = (int) Params[14];
  
  dim3 tpP(NchanNear, NrankPC);
        
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);

   /* collect input GPU variables*/
  mxGPUArray const  *W,  *data, *iC;
  mxGPUArray *featPC;
  float *d_featPC;
  const float     *d_W, *d_data;
  const int       *d_iC;
  
  data       = mxGPUCreateFromMxArray(prhs[1]);
  d_data     = (float const *)(mxGPUGetDataReadOnly(data));
  W             = mxGPUCreateFromMxArray(prhs[2]);
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  iC       = mxGPUCopyFromMxArray(prhs[3]);
  d_iC     = (int const *)(mxGPUGetDataReadOnly(iC));  
  
  /* allocate new GPU variables*/  
  float *d_dfilt;
  int *d_st,  *d_id, *d_counter;

  
  //cudaMalloc(&d_dout,   NT * Nchan* sizeof(float));
  cudaMalloc(&d_dfilt,  NT * Nchan* sizeof(float));
  cudaMalloc(&d_st,     maxFR * sizeof(int));
  cudaMalloc(&d_id,     maxFR * sizeof(int));
  cudaMalloc(&d_counter,   2*sizeof(int));

  cudaMemset(d_counter, 0, 2*sizeof(int));
  cudaMemset(d_dfilt,   0, NrankPC*NT * Nchan * sizeof(float));
  cudaMemset(d_st,      0, maxFR *   sizeof(int));
  cudaMemset(d_id,      0, maxFR *   sizeof(int));
    
  int *counter;
  counter = (int*) calloc(1,sizeof(int));
  
  // get the max of the data
  max1D<<<Nchan, Nthreads>>>(d_Params, d_data, d_dfilt);
  
  // take max across nearby channels
  maxChannels<<<NT/Nthreads,Nthreads>>>(d_Params, d_data, d_dfilt, d_iC, d_st, d_id, d_counter);
   
  counter[0] = min(maxFR, counter[0]);
  
  // move d_x to the CPU
  int minSize;
  if (counter[0]<maxFR)  minSize = counter[0];
  else                   minSize = maxFR;
//   const mwSize dimst[] 	= {minSize,1}; 
//   plhs[0] = mxCreateNumericArray(2, dimst, mxSINGLE_CLASS, mxREAL);
//   x =  (float*) mxGetData(plhs[0]);  
//   cudaMemcpy(x,    d_x, minSize * sizeof(float), cudaMemcpyDeviceToHost);
  
  // from each detected spike, project onto wPCA and extract those coefficients in iC channel order
  const mwSize dimsdWU[] 	= {NrankPC, NchanNear, minSize};
  featPC 		= mxGPUCreateGPUArray(3, dimsdWU, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  d_featPC 		= (float *)(mxGPUGetData(featPC));
  cudaMemset(d_featPC,    0, NrankPC*NchanNear*minSize*sizeof(float));
  
  computeProjections<<<minSize, tpP>>>(d_Params, d_data, d_iC, d_st, d_id, d_W, d_featPC);
  
  // d_feat is NrankPC by NchanNear by minsize and it comes out 
  
  
  // dWU stays a GPU array
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(featPC);  

  
  cudaFree(d_st);
  cudaFree(d_id);
  cudaFree(d_counter);
  cudaFree(d_Params); 
  cudaFree(d_dfilt);
  cudaFree(d_featPC);
  
  mxGPUDestroyGPUArray(W);  
  mxGPUDestroyGPUArray(iC);
  mxGPUDestroyGPUArray(data);  
}