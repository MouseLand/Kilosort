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

const int  Nthreads = 1024, maxFR = 5000, NrankMax = 6;
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  sumChannels(const double *Params, const float *data, 
	float *datasum, int *kkmax, const int *iC){
    
  int tid, tid0,t, kmax, i, bid, NT, Nchan, NchanNear,j,iChan, Nsum, Nrank;
  float  Cf, Cmax;
 
  NchanNear = (int) Params[10];  
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT 		= (int) Params[0];
  Nchan     = (int) Params[9];
  Nsum      = (int) Params[13];
  Nrank     = (int) Params[14];
  
  tid0 = tid + bid * blockDim.x;
  while (tid0<NT){
      for (i=0; i<Nchan;i++){
          Cmax = 0.0f;
          kmax = 0;
          for (t=0;t<Nrank;t++){ 
              Cf = 0.0f;              
              for(j=0; j<Nsum; j++){
                  iChan = iC[j+ NchanNear * i];
                  Cf    += data[tid0 + NT * iChan + t * NT * Nchan];
                  if (Cf*Cf/(1+j) > Cmax){
                      Cmax = Cf*Cf /(1+j);
                      kmax = j + t*Nsum;
                   }              
              }
          }
          datasum[tid0 + NT * i] = Cmax;
          kkmax[tid0 + NT * i]   = kmax;          
      }
      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	Conv1D(const double *Params, const float *data, const float *W, float *conv_sig){
    volatile __shared__ float  sW[81*NrankMax], sdata[(Nthreads+81)];
    float y;
    int tid, tid0, bid, i, nid, Nrank, NT, nt0,  Nchan;

    tid 		= threadIdx.x;
    bid 		= blockIdx.x;
    NT      	=   (int) Params[0];
    Nrank     = (int) Params[14];
    nt0       = (int) Params[4];
    Nchan     = (int) Params[9];
    
    if(tid<nt0*Nrank)
        sW[tid]= W[tid];
    __syncthreads();
    
    tid0 = 0;
    while (tid0<NT-Nthreads-nt0+1){
        if (tid<nt0)
            sdata[tid] = data[tid0 + tid + NT*bid];        
        sdata[tid + nt0] = data[nt0+tid0 + tid+ NT*bid];
        __syncthreads();
                
        for(nid=0;nid<Nrank;nid++){
            y = 0.0f;
            #pragma unroll 4
            for(i=0;i<nt0;i++)
                y    += sW[i + nid*nt0] * sdata[i+tid];                        
            conv_sig[tid0  + tid + NT*bid + nid * NT * Nchan]   = y;
        }
        tid0+=Nthreads;
        __syncthreads();
    }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  bestFilter(const double *Params, const float *data, 
	float *err, int *ftype, int *kkmax, int *kall){
    
  int tid, tid0, i, bid, NT, Nchan, ibest = 0, kbest;
  float  Cf, Cbest = 0.0f;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT 		= (int) Params[0];
  Nchan     = (int) Params[9];
  
  tid0 = tid + bid * blockDim.x;
  while (tid0<NT){
      kbest = 0;      
      for (i=0; i<Nchan;i++){
          Cf = data[tid0 + NT*i];
          if (Cf > Cbest + 1e-6){
              Cbest 	= Cf;
              ibest 	= i;
              kbest 	= kkmax[tid0 + NT*i];
          }          
      }
      err[tid0] 	= Cbest;
      ftype[tid0] 	= ibest;
      kall[tid0] 	= kbest;
      
      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	cleanup_spikes(const double *Params, const float *err, 
	const int *ftype, float *x, int *st, int *id, int *counter){
    
  int lockout, indx, tid, bid, NT, tid0,  j, t0;
  volatile __shared__ float sdata[Nthreads+2*81+1];
  bool flag=0;
  float err0, Th;
  
  lockout   = (int) Params[4] - 1;
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  NT      	=   (int) Params[0];
  tid0 		= bid * blockDim.x ;
  Th 		= (float) Params[2];
  //Th = 14.0f;
  
  while(tid0<NT-Nthreads-lockout+1){
      if (tid<2*lockout)
          sdata[tid] = err[tid0 + tid];
      if (tid0+tid+2*lockout<NT)
          sdata[tid+2*lockout] = err[2*lockout + tid0 + tid];
      else
          sdata[tid+2*lockout] = 0.0f;
      
      __syncthreads();
      
      err0 = sdata[tid+lockout];
      t0 = tid+lockout         + tid0;
      if(err0 > Th*Th && t0<NT-lockout-1){
          flag = 0;
          for(j=-lockout;j<=lockout;j++)
              if(sdata[tid+lockout+j]>err0){
                  flag = 1;
                  break;
              }
          if(flag==0){
              indx = atomicAdd(&counter[0], 1);
              if (indx<maxFR){
                  st[indx] = t0;
                  id[indx] = ftype[t0];
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
    
  int indx, tid, bid, t, d, Nmax;
  volatile __shared__ float s_id[maxFR], s_x[maxFR];
  bool flag=0;
  float xmax;
  
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
    
  Nmax = min(maxFR, counter[0]);
  
  while (tid<Nmax){
      s_x[tid]  = x[tid];
      s_id[tid] = id[tid];
      tid+=blockDim.x;
  }
  __syncthreads();
   
  tid = bid*blockDim.x + threadIdx.x;
          
  if (tid<Nmax){
      xmax = s_x[tid];
      flag = 1;
      for (t=0; t<Nmax;t++){
          d = abs(s_id[t] - s_id[tid]);
          if (d<5 && xmax< s_x[t]){
              flag = 0;
                break;
          }   
      }
      // if flag, then your thread is the max across nearby channels
      if(flag){
          indx = atomicAdd(&counter[1], 1);
          if (indx<maxFR){
              st1[indx] = st[tid];
              id1[indx] = s_id[tid];
          }
      }
  }
  
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void extract_snips(const double *Params, const int *st, const int *id,
        const int *counter, const float *dataraw,  float *WU){
    
  int nt0, tidx, tidy, bid, ind, NT, Nchan, Nmax;
  
  NT        = (int) Params[0];
  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
   
  tidx 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  Nmax = min(maxFR, counter[1]);
  
  for(ind=0; ind<Nmax;ind++)
      if (id[ind]==bid){
		  tidy 		= threadIdx.y;
		  while (tidy<Nchan){	
            WU[tidx+tidy*nt0 + nt0*Nchan * ind] = dataraw[st[ind]+tidx + NT * tidy];
			tidy+=blockDim.y;
		  }
	  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void extract_snips2(const double *Params, const float *err, const int *st, const int *id,
        const int *counter, const int *kk, const int *iC, const float *W, float *WU){
    
  int nt0, tidx, tidy, bid, ind, icl, Nchan, Nmax, Nsum, NchanNear;
  
  //NT        = (int) Params[0];
  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
  Nsum      = (int) Params[13];
  NchanNear = (int) Params[10];  
          
  tidx 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  Nmax = min(maxFR, counter[1]);
  
  for(ind=0; ind<Nmax;ind++)
      if (id[ind]==bid){
		  tidy 		= threadIdx.y;          
          icl = kk[st[ind]];
          
		  while (tidy<(1+icl%Nsum)){	
            //WU[tidx+tidy*nt0 + nt0*Nchan * ind] = dataraw[st[ind]+tidx + NT * tidy];
              WU[tidx + iC[tidy + bid*NchanNear]*nt0 + nt0*Nchan*ind] = 
                      sqrt(err[st[ind]] / (1.+icl%Nsum)) * W[tidx + nt0 * int(icl/Nsum)];
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
  unsigned int nt0, NT, Nchan, Nnearest, Nrank;
  
  /* read Params and copy to GPU */
  Params  	= (double*) mxGetData(prhs[0]);
  NT		= (unsigned int) Params[0];
  Nchan     = (unsigned int) Params[9];
  nt0       = (unsigned int) Params[4];
  Nnearest  = (unsigned int) Params[5];
  Nrank     = (unsigned int) Params[14];
  
  dim3 tpB(8, 2*nt0-1), tpF(16, Nnearest), tpS(nt0, 16);
        
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);

   /* collect input GPU variables*/
  mxGPUArray const  *W,  *data, *iC;
  const float     *d_W, *d_data;
  const int       *d_iC;
  
  data       = mxGPUCreateFromMxArray(prhs[1]);
  d_data     = (float const *)(mxGPUGetDataReadOnly(data));
  W             = mxGPUCreateFromMxArray(prhs[2]);
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  iC       = mxGPUCopyFromMxArray(prhs[3]);
  d_iC     = (int const *)(mxGPUGetDataReadOnly(iC));  
  
  /* allocate new GPU variables*/  
  float *d_err, *d_x, *d_dout, *d_WU, *d_dfilt;
  int *d_st, *d_kkmax,*d_kk, *d_id1, *d_st1, *d_ftype,  *d_id, *d_counter;

  mxGPUArray *dout;
  const mwSize dimst[] 	= {NT, Nchan};
  dout 		= mxGPUCreateGPUArray(2, dimst, mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);  
  d_dout 		= (float *)(mxGPUGetData(dout));  
  
  //cudaMalloc(&d_dout,   NT * Nchan* sizeof(float));
  cudaMalloc(&d_dfilt,  Nrank*NT * Nchan* sizeof(float));
  cudaMalloc(&d_err,    NT * sizeof(float));
  cudaMalloc(&d_kkmax,  Nchan*NT * sizeof(int));
  cudaMalloc(&d_kk,     NT * sizeof(int));
  cudaMalloc(&d_ftype,  NT * sizeof(int));  
  cudaMalloc(&d_st,     maxFR * sizeof(int));
  cudaMalloc(&d_id,     maxFR * sizeof(int));
  cudaMalloc(&d_x,      maxFR * sizeof(float));  
  cudaMalloc(&d_st1,    maxFR * sizeof(int));
  cudaMalloc(&d_id1,    maxFR * sizeof(int));
  cudaMalloc(&d_counter,   2*sizeof(int));

  cudaMemset(d_counter, 0, 2*sizeof(int));
  cudaMemset(d_dout,    0, NT * Nchan * sizeof(float));
  cudaMemset(d_dfilt,   0, Nrank*NT * Nchan * sizeof(float));
  cudaMemset(d_err,     0, NT * sizeof(float));
  cudaMemset(d_kkmax,   0, NT * Nchan* sizeof(int));
  cudaMemset(d_kk,      0, NT * sizeof(int));
  cudaMemset(d_ftype,   0, NT * sizeof(int));
  cudaMemset(d_st,      0, maxFR *   sizeof(int));
  cudaMemset(d_id,      0, maxFR *   sizeof(int));
  cudaMemset(d_x,       0, maxFR *   sizeof(float));
  cudaMemset(d_st1,     0, maxFR *   sizeof(int));
  cudaMemset(d_id1,     0, maxFR *   sizeof(int));
    
  unsigned int *counter;
  counter = (unsigned int*) calloc(1,sizeof(unsigned int));
  
  // filter the data with the temporal templates
  Conv1D<<<Nchan, Nthreads>>>(d_Params, d_data, d_W, d_dfilt);
  
  // sum each template across channels, square, take max
  sumChannels<<<NT/Nthreads,Nthreads>>>(d_Params, d_dfilt, d_dout, d_kkmax, d_iC);
   
  // compute the best filter
  bestFilter<<<NT/Nthreads,Nthreads>>>(d_Params, d_dout, d_err, d_ftype, d_kkmax, d_kk);
 
  // ignore peaks that are smaller than another nearby peak
  cleanup_spikes<<<NT/Nthreads,Nthreads>>>(d_Params,
          d_err, d_ftype, d_x, d_st, d_id, d_counter); // NT/Nthreads 
   
  // ignore peaks that are smaller than another nearby peak
  cleanup_heights<<<1 + maxFR/32 , 32>>>(d_Params, d_x, d_st, d_id, d_st1, d_id1, d_counter); // 1 + maxFR/32
   
  // add new spikes to 2nd counter
  cudaMemcpy(counter,     d_counter+1, sizeof(int), cudaMemcpyDeviceToHost);
  
  counter[0] = min(maxFR, counter[0]);
  
  cudaMalloc(&d_WU,    counter[0]*nt0*Nchan * sizeof(float));  
  cudaMemset(d_WU,      0, counter[0]*nt0*Nchan * sizeof(float));
    
  mxGPUArray *WU1;
  float  *d_WU1;  
  const mwSize dimsu[] 	= {nt0, Nchan, counter[0]};  
  WU1 		= mxGPUCreateGPUArray(3, dimsu, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
  d_WU1 		= (float *)(mxGPUGetData(WU1));
  
  // update dWU here by adding back to subbed spikes
  //extract_snips2<<<Nchan,tpS>>>(  d_Params, d_err, d_st1, d_id1, d_counter, d_kk, d_iC, d_W, d_WU);
  extract_snips<<<Nchan,tpS>>>(  d_Params, d_st1, d_id1, d_counter, d_data, d_WU);

  if (counter[0]>0)
      cudaMemcpy(d_WU1, d_WU, nt0*Nchan*counter[0]*sizeof(float), cudaMemcpyDeviceToDevice);  
  
  // dWU stays a GPU array
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(WU1);  
  plhs[1] 	= mxGPUCreateMxArrayOnGPU(dout); 

  
  cudaFree(d_ftype);
  cudaFree(d_kkmax);
  cudaFree(d_err);
  cudaFree(d_st);
  cudaFree(d_id);
  cudaFree(d_st1);
  cudaFree(d_x);
  cudaFree(d_kk);
  cudaFree(d_id1);
  cudaFree(d_counter);
  cudaFree(d_Params); 
  cudaFree(d_dfilt);
  cudaFree(d_WU);
  
  mxGPUDestroyGPUArray(W);  
  mxGPUDestroyGPUArray(iC);
  mxGPUDestroyGPUArray(data);
  mxGPUDestroyGPUArray(WU1);  
  mxGPUDestroyGPUArray(dout);
}