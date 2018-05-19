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

const int  Nthreads = 1024, maxFR = 500, NrankMax = 3;
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	Conv1D(const double *Params, const float *data, const float *W, float *conv_sig){    
  volatile __shared__ float  sW[81*NrankMax], sdata[(Nthreads+81)*NrankMax]; 
  float x, y;
  int tid, tid0, bid, i, nid, Nrank, NT, nt0;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT      	=   (int) Params[0];
  Nrank     = (int) Params[6];
  nt0       = (int) Params[4];
   
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
          
          if (nid==0 && y<0)
              break;

           x += y*y;
      }
      conv_sig[tid0  + tid + NT*bid]   = x;
      
      tid0+=Nthreads;
      __syncthreads();
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  bestFilter(const double *Params, const float *data, 
	float *err, int *ftype){
    
  int tid, tid0, i, bid, NT, Nchan, ibest = 0;
  float  Cf, Cbest = 0.0f;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT 		= (int) Params[0];
  Nchan     = (int) Params[9];
  
  tid0 = tid + bid * blockDim.x;
  while (tid0<NT){
      for (i=0; i<Nchan;i++){
          Cf = data[tid0 + NT * i];
          
          if (Cf > Cbest + 1e-6){
              Cbest 	= Cf;
              ibest 	= i;
          }
      }
      err[tid0] 	= Cbest;
      ftype[tid0] 	= ibest;
      
      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	cleanup_spikes(const double *Params, const float *err, 
	const int *ftype, float *x, int *st, int *id, int *counter){
    
  int lockout, indx, tid, bid, NT, tid0,  j;
  volatile __shared__ float sdata[Nthreads+2*81+1];
  bool flag=0;
  float err0, Th;
  
  lockout   = (int) Params[4] - 1;
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  NT      	=   (int) Params[0];
  tid0 		= bid * blockDim.x ;
  Th 		= (float) Params[2];
  
  while(tid0<NT-Nthreads-lockout+1){
      if (tid<2*lockout)
          sdata[tid] = err[tid0 + tid];
      sdata[tid+2*lockout] = err[2*lockout + tid0 + tid];
      
      __syncthreads();
      
      err0 = sdata[tid+lockout];
      if(err0>2*Th*Th){
          flag = 0;
          for(j=-lockout;j<=lockout;j++)
              if(sdata[tid+lockout+j]>err0){
                  flag = 1;
                  break;
              }
          if(flag==0){
              indx = atomicAdd(&counter[0], 1);
              if (indx<maxFR){
                  st[indx] = tid+lockout         + tid0;
                  id[indx] = ftype[tid+lockout   + tid0];
                  x[indx]  = err0;
              }
          }
      }
      
      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	cleanup_heights(const double *Params, const float *x, 
        const int *st, const int *id, int *st1, int *id1, int *counter){
    
  int indx, tid, bid, t, d;
  volatile __shared__ float s_id[maxFR], s_x[maxFR];
  bool flag=0;
  float xmax;
  
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
    
  while (tid<counter[0]){
      s_x[tid]  = x[tid];
      s_id[tid] = id[tid];
      tid+=blockDim.x;
  }
  __syncthreads();
   
  tid = bid*blockDim.x + threadIdx.x;
          
  if (tid<counter[0]){
      xmax = s_x[tid];
      flag = 1;
      for (t=0; t<counter[0];t++){
          d = abs(s_id[t] - s_id[tid]);
          if (d<5 && xmax< s_x[t]){
              flag = 0;
                break;
          }   
      }
      // if flag, then your thread is the max across nearby channels
      if(flag){
          indx = atomicAdd(&counter[1], 1);
          st1[indx] = st[tid];
          id1[indx] = s_id[tid];
      }
  }
  
}


//////////////////////////////////////////////////////////////////////////////////////////
__global__ void extract_snips(const double *Params, const int *st, const int *id,
        const int *counter, const float *dataraw,  float *WU){
    
  int nt0, tidx, tidy, bid, ind, NT, Nchan;
  
  NT        = (int) Params[0];
  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
   
  tidx 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  for(ind=0; ind<counter[1];ind++)
      if (id[ind]==bid){
		  tidy 		= threadIdx.y;
		  while (tidy<Nchan){	
            WU[tidx+tidy*nt0 + nt0*Nchan * ind] = dataraw[st[ind]+tidx + NT * tidy];
			tidy+=blockDim.y;
		  }
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
  int nt0, NT, Nchan, Nnearest;

  
  /* read Params and copy to GPU */
  Params  	= (double*) mxGetData(prhs[0]);
  NT		= (int) Params[0];
  Nchan     = (int) Params[9];
  nt0       = (int) Params[4];
  Nnearest  = (int) Params[5];
  
  dim3 tpB(8, 2*nt0-1), tpF(16, Nnearest), tpS(nt0, 16);
        
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);

   /* collect input GPU variables*/
  mxGPUArray const  *W,  *data;
  const float     *d_W, *d_data;
  
  data       = mxGPUCreateFromMxArray(prhs[1]);
  d_data     = (float const *)(mxGPUGetDataReadOnly(data));
  W             = mxGPUCreateFromMxArray(prhs[2]);
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  
  
  /* allocate new GPU variables*/  
  float *d_err,*d_x, *d_dout, *d_WU;
  int *d_st, *d_id1, *d_st1, *d_ftype,  *d_id, *d_counter;

  cudaMalloc(&d_dout,   NT * Nchan* sizeof(float));
  cudaMalloc(&d_err,   NT * sizeof(float));
  cudaMalloc(&d_ftype, NT * sizeof(int));  
  cudaMalloc(&d_st,    maxFR * sizeof(int));
  cudaMalloc(&d_id,    maxFR * sizeof(int));
  cudaMalloc(&d_st1,    maxFR * sizeof(int));
  cudaMalloc(&d_id1,    maxFR * sizeof(int));
  cudaMalloc(&d_x,     maxFR * sizeof(float));
  
  cudaMalloc(&d_WU,    maxFR*nt0*Nchan * sizeof(float));
  cudaMalloc(&d_counter,   2*sizeof(int));
  
  cudaMemset(d_WU,      0, maxFR*nt0*Nchan * sizeof(float));
  cudaMemset(d_counter, 0, 2*sizeof(int));
  cudaMemset(d_dout,    0, NT * Nchan * sizeof(float));
  cudaMemset(d_err,     0, NT * sizeof(float));
  cudaMemset(d_ftype,   0, NT * sizeof(int));
  cudaMemset(d_st,      0, maxFR *   sizeof(int));
  cudaMemset(d_id,      0, maxFR *   sizeof(int));
  cudaMemset(d_st1,      0, maxFR *   sizeof(int));
  cudaMemset(d_id1,      0, maxFR *   sizeof(int));
  cudaMemset(d_x,      0, maxFR *   sizeof(float));
  
  
  int *counter;
  counter = (int*) calloc(1,sizeof(int));
  
  // filter the data with the temporal templates
  Conv1D<<<Nchan, Nthreads>>>(d_Params, d_data, d_W, d_dout);
  
  // compute the best filter
  bestFilter<<<NT/Nthreads,Nthreads>>>(d_Params, d_dout, d_err, d_ftype);
  
  // ignore peaks that are smaller than another nearby peak
  cleanup_spikes<<<NT/Nthreads,Nthreads>>>(d_Params,
          d_err, d_ftype, d_x, d_st, d_id, d_counter);
  
  // ignore peaks that are smaller than another nearby peak
  cleanup_heights<<<1 + maxFR/32, 32>>>(d_Params, d_x, d_st, d_id, d_st1, d_id1, d_counter);
  
  // add new spikes to 2nd counter
  cudaMemcpy(counter,     d_counter+1, sizeof(int), cudaMemcpyDeviceToHost);
  
  // update dWU here by adding back to subbed spikes
  extract_snips<<<Nchan,tpS>>>(  d_Params, d_st1, d_id1, d_counter, d_data, d_WU);
  
  mxGPUArray *WU1;
  float  *d_WU1;
  const mwSize dimsu[] 	= {nt0, Nchan, counter[0]};
  WU1 		= mxGPUCreateGPUArray(3, dimsu, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
  d_WU1 		= (float *)(mxGPUGetData(WU1));
  
  cudaMemcpy(d_WU1, d_WU, nt0*Nchan*counter[0]*sizeof(float), cudaMemcpyDeviceToDevice);
  
  // dWU stays a GPU array
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(WU1);

  
  cudaFree(d_ftype);
  cudaFree(d_err);
  cudaFree(d_st);
  cudaFree(d_id);
  cudaFree(d_st1);
  cudaFree(d_x);
  cudaFree(d_id1);
  cudaFree(d_counter);
  cudaFree(d_Params);
  
  cudaFree(d_dout);
  cudaFree(d_WU);
  
  mxGPUDestroyGPUArray(W);
  mxGPUDestroyGPUArray(data);
  mxGPUDestroyGPUArray(WU1);
}