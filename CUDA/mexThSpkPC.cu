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

const int  Nthreads = 1024, maxFR = 100000, NrankMax = 3, nt0max=81, NchanMax = 17;

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	Conv1D(const double *Params, const float *data, const float *W, float *conv_sig){    
  volatile __shared__ float  sW[81*NrankMax], sdata[Nthreads+81]; 
  float x, y;
  int tid, tid0, bid, i, nid, Nrank, NT, nt0;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  NT      	=   (int) Params[0];
  nt0       = (int) Params[3];
  Nrank     = (int) Params[6];  
   
  if(tid<nt0*Nrank)
      sW[tid]= W[tid];
  __syncthreads();
  
  tid0 = 0;
  while (tid0<NT-Nthreads-nt0+1){
	  if (tid<nt0) 
          sdata[tid] = data[tid0 + tid+ NT*bid];
	  
      sdata[tid + nt0] = data[tid0 + tid + nt0 + NT*bid];	  
	  __syncthreads();
      
	  x = 0.0f;
      for(nid=0;nid<Nrank;nid++){
          y = 0.0f;
		  #pragma unroll 4
          for(i=0;i<nt0;i++)
              y    += sW[i + nid*nt0] * sdata[i+tid];

           x += y*y;
      }
      conv_sig[tid0  + tid + NT*bid]   = sqrt(x);
      
      tid0+=Nthreads;
      __syncthreads();
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  computeProjections(const double *Params, const float *dataraw,
        const int *iC, const int *st, const int *id, const float *W, float *feat){
            
    //number of blocks = number of spikes to process minimum( number found, maxFR=100000)
    //Thread grid = (NchanNear, NrankPC)
            
    float x;
    int tidx, nt0min, tidy, my_chan, this_chan, tid, bid, nt0, NchanNear, j, t, NT, NrankPC;
    volatile __shared__ float sW[nt0max*NrankMax], sD[nt0max*NchanMax];
    
    NT 		= (int) Params[0];    
    NchanNear = (int) Params[2];
    nt0       = (int) Params[3];        
    NrankPC  = (int) Params[6];
    nt0min    = (int) Params[4];
    
    tidx = threadIdx.x;        //PC index in W (column index)
    tidy = threadIdx.y;        //channel index
    bid = blockIdx.x;          //NchanNear*NrankPC; each spike gets NchanNear*NrankPC values in projection
    
    // move wPCA to shared memory
    while (tidx<nt0){
        sW[tidx + tidy*nt0] = W[tidx + tidy*nt0];
        tidx+=blockDim.x;
    }
    tidx = threadIdx.x;
    
    tid = tidx + tidy*blockDim.x;
    // move raw data to shared memory    
    while (tid<nt0){
        my_chan = id[bid];
        for (j=0;j<NchanNear;j++){
            this_chan = iC[j + NchanNear*my_chan];
            sD[tid + nt0*j] = dataraw[tid + st[bid]+nt0min-1 + NT * this_chan];
        }
        tid+=blockDim.x*blockDim.y;
    }
    __syncthreads();
    
    x = 0.0f;
    for (t=0;t<nt0;t++)
        x += sD[t + nt0*tidx] * sW[t + nt0*tidy];
                
    feat[tidy + tidx*NrankPC + NrankPC*NchanNear*bid] = x;
    
    }

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  maxChannels(const double *Params, const float *dataraw, const float *data,
	const int *iC, int *st, int *id, int *counter){
  
  //NT/Nthreads blocks, Nthreads threads
  //dataraw = data convolved with templates
  //data = max1D output = each data point replaced by the max value within nt0 points

  int nt0, indx, tid, tid0, i, bid, NT, Nchan, NchanNear,j,iChan, nt0min;
  double Cf, d;
  float spkTh;
  bool flag;

  NT 		= (int) Params[0];
  Nchan     = (int) Params[1];  
  NchanNear = (int) Params[2];      
  nt0       = (int) Params[3];    
  nt0min    = (int) Params[4];
  spkTh    = (float) Params[5];  
  
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  tid0 = tid + bid * blockDim.x;
  while (tid0<NT-nt0-nt0min){
      for (i=0; i<Nchan;i++){          
          iChan = iC[0 + NchanNear * i];
          Cf    = (double) data[tid0 + NT * iChan];
          flag = true;
          
          for(j=1; j<NchanNear; j++){
              iChan = iC[j+ NchanNear * i];              
              if (data[tid0 + NT * iChan] > Cf){                
                flag = false;
                break;
              }                
          }
          
          if (flag){
              iChan = iC[NchanNear * i];
              if (Cf>spkTh){
                  d = (double) dataraw[tid0+nt0min-1 + NT*iChan]; // 
                  if (d > Cf-1e-6){
                      // this is a hit, atomicAdd and return spikes
                      indx = atomicAdd(&counter[0], 1);
                      if (indx<maxFR){
                          st[indx] = tid0;
                          id[indx] = iChan;
                      }
                  }
              }
          }          
      }
      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	max1D(const double *Params, const float *data, float *conv_sig){
    
    volatile __shared__ float  sdata[Nthreads+81];
    float y, spkTh;
    int tid, tid0, bid, i, NT, nt0;
    
    NT 		= (int) Params[0];        
    nt0       = (int) Params[3];    
    spkTh    = (float) Params[5];    
    tid 		= threadIdx.x;
    bid 		= blockIdx.x;
  
    tid0 = 0;
    while (tid0<NT-Nthreads-nt0+1){
        if (tid<nt0)
            sdata[tid]   = data[tid0 + tid + NT*bid];
        sdata[tid + nt0] = data[nt0+tid0 + tid+ NT*bid];
        __syncthreads();

        y = 0.0f;
        #pragma unroll 4
        for(i=0;i<nt0;i++)
            y    = max(y, sdata[tid+i]);
        
        if (y>spkTh)
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
  unsigned int NT, Nchan, NchanNear, NrankPC;
  
  /* read Params and copy to GPU */
  Params  	= (double*) mxGetData(prhs[0]);
  NT 		= (unsigned int) Params[0];
  Nchan     = (unsigned int) Params[1];  
  NchanNear = (unsigned int) Params[2];      
  NrankPC     = (unsigned int) Params[6];
        
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);

   /* collect input GPU variables*/
  mxGPUArray const  *W,  *data, *iC;
  mxGPUArray *featPC, *id;
  float *d_featPC;
  int *d_id2;
  const float     *d_W, *d_data;
  const int       *d_iC;
  
  data       = mxGPUCreateFromMxArray(prhs[1]);
  d_data     = (float const *)(mxGPUGetDataReadOnly(data));
  W             = mxGPUCreateFromMxArray(prhs[2]);
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  iC       = mxGPUCopyFromMxArray(prhs[3]);
  d_iC     = (int const *)(mxGPUGetDataReadOnly(iC));  
  
  /* allocate new GPU variables*/  
  float *d_dmax, *d_dout;
  int *d_st,  *d_id, *d_counter;
  
  cudaMalloc(&d_dout,   NT * Nchan* sizeof(float));
  cudaMalloc(&d_dmax,  NT * Nchan* sizeof(float));
  cudaMalloc(&d_st,     maxFR * sizeof(int));
  cudaMalloc(&d_id,     maxFR * sizeof(int));
  cudaMalloc(&d_counter,   2*sizeof(int));

  cudaMemset(d_dout,   0, NT * Nchan* sizeof(float)); 
  cudaMemset(d_dmax,   0, NT * Nchan * sizeof(float));
  cudaMemset(d_st,      0, maxFR *   sizeof(int));
  cudaMemset(d_id,      0, maxFR *   sizeof(int));
  cudaMemset(d_counter, 0, 2*sizeof(int));
     
  int *counter;
  counter = (int*) calloc(1,sizeof(int));
  
  // filter the data with the temporal templates
  Conv1D<<<Nchan, Nthreads>>>(d_Params, d_data, d_W, d_dout);
  
  // get the max of the data
  max1D<<<Nchan, Nthreads>>>(d_Params, d_dout, d_dmax);
  
  // take max across nearby channels
  // return spike times in d-st, max channel index in d_id, #spikes in d_counter
  // note that max channel and spike times are only saved for the first maxFR spikes
  maxChannels<<<NT/Nthreads,Nthreads>>>(d_Params, d_dout, d_dmax, d_iC, d_st, d_id, d_counter);
  
  cudaMemcpy(counter,     d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  
  // calculate features for up to maxFR spikes 
  unsigned int minSize=1;
  minSize = min(maxFR, counter[0]);

  const mwSize ddF[] 	= {NrankPC * NchanNear, minSize};
  featPC 		= mxGPUCreateGPUArray(2, ddF, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  d_featPC 		= (float *)(mxGPUGetData(featPC));
  cudaMemset(d_featPC, 0, NrankPC*NchanNear*minSize*sizeof(float));
      
  const mwSize did[] 	= {minSize, 1};
  id 		= mxGPUCreateGPUArray(2, did, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  d_id2 		= (int *)(mxGPUGetData(id));
  
  dim3 tpP(NchanNear, NrankPC);
  if (minSize>0)      
      computeProjections<<<minSize, tpP>>>(d_Params, d_data, d_iC, d_st, d_id, d_W, d_featPC);  
  
  cudaMemcpy(d_id2, d_id, minSize * sizeof(int),   cudaMemcpyDeviceToDevice);

  
  
  // uproj and array of max channels will remain GPU arrays
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(featPC);  
  plhs[1] 	= mxGPUCreateMxArrayOnGPU(id);  

  cudaFree(d_st);
  cudaFree(d_id);  
  cudaFree(d_counter);
  cudaFree(d_Params); 
  cudaFree(d_dmax);
  cudaFree(d_dout);  
  
  mxGPUDestroyGPUArray(featPC);  
  mxGPUDestroyGPUArray(id);  
  mxGPUDestroyGPUArray(W);  
  mxGPUDestroyGPUArray(iC);
  mxGPUDestroyGPUArray(data);  
}