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

const int  Nthreads = 1024,  NrankMax = 6, maxFR = 10000, nt0max=81, NchanMax = 17, nsizes = 5;


//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	Conv1D(const double *Params, const float *data, const float *W, float *conv_sig){
    volatile __shared__ float  sW[81*NrankMax], sdata[(Nthreads+81)];
    float y;
    int tid, tid0, bid, i, nid, Nrank, NT, nt0,  Nchan;

    tid 		= threadIdx.x;
    bid 		= blockIdx.x;
    
    NT        = (int) Params[0];
    Nchan     = (int) Params[1];
    nt0       = (int) Params[2];
    Nrank     = (int) Params[4];
    
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
__global__ void  sumChannels(const double *Params, const float *data, 
	float *datasum, int *kkmax, const int *iC2, const float *dist, const float *v2){
    
  int tid, tid0,t,k, kmax, bidx, bidy, NT, Nchan, NchanNear,j,iChan, Nsum, Nrank;
  float  Cmax, C0;
  float a[nsizes], d2;
  float  sigma;
  volatile __shared__ float  sA[nsizes * 20];
  
  
  tid 		= threadIdx.x;
  bidx 		= blockIdx.x;
  bidy 		= blockIdx.y;
  NT 		= (int) Params[0];
  Nchan     = (int) Params[1];
  NchanNear = (int) Params[3];  
  Nrank     = (int) Params[4];
  Nsum      = (int) Params[3];
  sigma = (float) Params[9];
  
  if (tid<nsizes*NchanNear){
      d2 = dist[tid/nsizes + NchanNear * bidy];        
      k = tid%nsizes;
      sA[tid] = expf( - (d2 * d2)/((1+k)*(1+k)*sigma*sigma));
  }
  __syncthreads();
  
  tid0 = tid + bidx * blockDim.x;
  while (tid0<NT){
      Cmax = 0.0f;
      kmax = 0;
      
      for (t=0;t<Nrank;t++){                             
          for(k=0; k<nsizes; k++)
              a[k] = 0.;
                
          for(j=0; j<Nsum; j++){
              iChan = iC2[j + NchanNear * bidy];              
              for(k=0; k<nsizes; k++)
                  a[k]  += sA[k + nsizes * j] * 
                        data[tid0 + NT * iChan + t * NT * Nchan];
          }
          for(k=0; k<nsizes; k++){    
              a[k] = max(a[k], 0.);
              if (a[k]*a[k] / v2[k + nsizes*bidy] > Cmax){
                  Cmax = a[k]*a[k]/v2[k + nsizes*bidy];
                  kmax = t + k*Nrank;
               }             
          }
      }
      datasum[tid0 + NT * bidy] = Cmax;
      kkmax[tid0 + NT * bidy]   = kmax;          

      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	max1D(const double *Params, const float *data, float *conv_sig){
    
    volatile __shared__ float  sdata[Nthreads+81];
    float y, spkTh;
    int tid, tid0, bid, i, NT, nt0, nt0min;
    
    NT 		= (int) Params[0];        
    nt0       = (int) Params[2];        
    nt0min    = (int) Params[5];
    spkTh    = (float) Params[6];    
    
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
        for(i=0;i<2*nt0min;i++)
            y    = max(y, sdata[tid+i]);
        
        if (y>spkTh*spkTh)
            conv_sig[tid0 + 1*(nt0min) + tid + NT*bid]   = y;

        tid0+=Nthreads;
        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  maxChannels(const double *Params, const float *dataraw, const float *data,
	const int *iC,  const int *iC2, const float *dist2, const int *kkmax, 
        const float *dfilt, int *st, int *counter, float *cF){
    
  int nt0, indx, tid, tid0, i, bid, NT, j,iChan, nt0min, Nrank, kfilt;
  int Nchan, NchanNear, NchanUp, NchanNearUp, bidy ;
  double Cf, d;
  float spkTh, d2;
  bool flag;
 
  NT 		= (int) Params[0];
  Nchan     = (int) Params[1];  
  NchanNear = (int) Params[3];    
  NchanUp     = (int) Params[7];  
  NchanNearUp = (int) Params[8];    
  nt0       = (int) Params[2];      
  nt0min    = (int) Params[5];
  spkTh    = (float) Params[6];
  Nrank     = (int) Params[4];
  
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  bidy = blockIdx.y;
  
  tid0 = tid + bid * blockDim.x;
  while (tid0<NT-nt0-nt0min){
      i = bidy;
      Cf    = (double) data[tid0 + NT * i];
      flag = true;
      for(j=1; j<NchanNearUp; j++){
          if (dist2[j + NchanNearUp * i] < 100.){
              iChan = iC2[j+ NchanNearUp * i];
              if (data[tid0 + NT * iChan] > Cf){
                  flag = false;
                  break;
              }
          }
      }
      
      if (flag){
          if (Cf>spkTh*spkTh){
              d = (double) dataraw[tid0+0 * (nt0min-1) + NT*i]; //
              if (d > Cf-1e-6){
                  // this is a hit, atomicAdd and return spikes
                  indx = atomicAdd(&counter[0], 1);
                  if (indx<maxFR){
                      st[0+4*indx] = tid0;
                      st[1+4*indx] = i;
                      st[2+4*indx] = sqrt(d);
                      st[3+4*indx] = kkmax[tid0+0*(nt0min-1) + NT*i];
                      kfilt = st[3+4*indx]%Nrank;
                      for(j=0; j<NchanNear; j++){
                          iChan = iC[j+ NchanNear * i];
                          cF[j + NchanNear * indx] = dfilt[tid0+0*(nt0min-1) + NT * iChan + kfilt * Nchan*NT];
                      }
                  }
              }
          }
      }
      
      tid0 += blockDim.x * gridDim.x;
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
  unsigned int nt0, NT, Nchan, Nnearest, Nrank, NchanUp;
  
  /* read Params and copy to GPU */
  Params  	= (double*) mxGetData(prhs[0]);
  NT		= (unsigned int) Params[0]; // 0
  Nchan     = (unsigned int) Params[1]; // 9
  nt0       = (unsigned int) Params[2];  // 4
  Nnearest  = (unsigned int) Params[3]; // 5
  Nrank     = (unsigned int) Params[4]; // 14
  NchanUp     = (unsigned int) Params[7]; // 14
  
        
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);

   /* collect input GPU variables*/
  mxGPUArray const  *W,  *data, *iC, *iC2, *dist, *v2, *dist2;
  const float     *d_W, *d_data, *d_dist, *d_v2, *d_dist2;
  const int       *d_iC, *d_iC2;
  
  data       = mxGPUCreateFromMxArray(prhs[1]);
  d_data     = (float const *)(mxGPUGetDataReadOnly(data));
  W             = mxGPUCreateFromMxArray(prhs[2]);
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  iC       = mxGPUCopyFromMxArray(prhs[3]);
  d_iC     = (int const *)(mxGPUGetDataReadOnly(iC));    
  dist             = mxGPUCreateFromMxArray(prhs[4]);
  d_dist        	= (float const *)(mxGPUGetDataReadOnly(dist));
  v2             = mxGPUCreateFromMxArray(prhs[5]);
  d_v2        	= (float const *)(mxGPUGetDataReadOnly(v2));
  iC2       = mxGPUCopyFromMxArray(prhs[6]);
  d_iC2     = (int const *)(mxGPUGetDataReadOnly(iC2));  
  dist2       = mxGPUCopyFromMxArray(prhs[7]);
  d_dist2     = (float const *)(mxGPUGetDataReadOnly(dist2));  
  
  /* allocate new GPU variables*/  
  float *d_dout, *d_dfilt, *d_dmax, *d_cF;
  int *d_kkmax;
  int *d_st,  *d_counter;
  
  mxGPUArray *dout, *kkmax;
  const mwSize dimst[] 	= {NT, NchanUp};
  dout 		= mxGPUCreateGPUArray(2, dimst, mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);  
  d_dout 		= (float *)(mxGPUGetData(dout));    
  kkmax 		= mxGPUCreateGPUArray(2, dimst, mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);  
  d_kkmax 		= (int *)(mxGPUGetData(kkmax));  
  
  cudaMalloc(&d_dfilt,  NT * Nchan * Nrank   * sizeof(float));
  cudaMalloc(&d_dmax,  NT * NchanUp* sizeof(float));
  cudaMalloc(&d_st,     4*maxFR * sizeof(int));  
  cudaMalloc(&d_cF,     Nnearest*maxFR * sizeof(float));  
  cudaMalloc(&d_counter,   2*sizeof(int));
  
  cudaMemset(d_dout,    0, NT * NchanUp * sizeof(float));
  cudaMemset(d_dmax,   0, NT * NchanUp * sizeof(float));
  cudaMemset(d_kkmax,   0, NT * NchanUp * sizeof(int));
  cudaMemset(d_dfilt,   0, NT * Nchan * Nrank   * sizeof(float));  
  cudaMemset(d_st,      0, 4*maxFR *   sizeof(int));  
  cudaMemset(d_cF,      0, Nnearest*maxFR *   sizeof(float));  
  cudaMemset(d_counter, 0, 2*sizeof(int));
  
  // filter the data with the temporal templates
  Conv1D<<<Nchan, Nthreads>>>(d_Params, d_data, d_W, d_dfilt);
  
  // sum each template across channels, square, take max
  dim3 tpP(NT/Nthreads, NchanUp);
  sumChannels<<<tpP,Nthreads>>>(d_Params, d_dfilt, d_dout, d_kkmax, d_iC, d_dist, d_v2);
  
  // get the max of the data
  max1D<<<NchanUp, Nthreads>>>(d_Params, d_dout, d_dmax);
  
  // take max across nearby channels
  int *counter;
  counter = (int*) calloc(1,sizeof(int));  
  dim3 tpC(NT/Nthreads, NchanUp);
  maxChannels<<<tpC,Nthreads>>>(d_Params, d_dout, d_dmax, d_iC, d_iC2, d_dist2, d_kkmax,d_dfilt,  d_st, d_counter, d_cF);
  
  cudaMemcpy(counter,     d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  
  unsigned int minSize=1;
  int *d_sto;
  mxGPUArray *sto;
  minSize = min(maxFR, counter[0]);
  const mwSize did[] 	= {4, minSize};
  sto 		= mxGPUCreateGPUArray(2, did, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  d_sto 		= (int *)(mxGPUGetData(sto));
  cudaMemcpy(d_sto, d_st, 4*minSize * sizeof(int),   cudaMemcpyDeviceToDevice);
  
  float *d_cF2;
  mxGPUArray *cF2;
  const mwSize dcf[] 	= {Nnearest, minSize};
  cF2 		= mxGPUCreateGPUArray(2, dcf, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  d_cF2 		= (float *)(mxGPUGetData(cF2));
 
  cudaMemcpy(d_cF2, d_cF, Nnearest*minSize * sizeof(float),   cudaMemcpyDeviceToDevice);
  
// dWU stays a GPU array
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(dout);    
  plhs[1] 	= mxGPUCreateMxArrayOnGPU(kkmax);    
  plhs[2] 	= mxGPUCreateMxArrayOnGPU(sto);    
  plhs[3] 	= mxGPUCreateMxArrayOnGPU(cF2);    

  
  cudaFree(d_Params); 
  cudaFree(d_dfilt);
  cudaFree(d_st);
  cudaFree(d_cF);
  cudaFree(d_dmax);
  cudaFree(d_counter);

  mxGPUDestroyGPUArray(W);  
  mxGPUDestroyGPUArray(v2);  
  mxGPUDestroyGPUArray(dist);
  mxGPUDestroyGPUArray(dist2);
  mxGPUDestroyGPUArray(iC);
  mxGPUDestroyGPUArray(iC2);
  mxGPUDestroyGPUArray(data);  
  
  mxGPUDestroyGPUArray(dout);
  mxGPUDestroyGPUArray(kkmax);
  mxGPUDestroyGPUArray(sto);
  mxGPUDestroyGPUArray(cF2);
}