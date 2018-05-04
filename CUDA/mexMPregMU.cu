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

const int Nthreads = 1024, NchanMax = 128, NrankMax = 3;
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	Conv1D(const double *Params, const float *data, const float *W, float *conv_sig){    
  __shared__ float  sW[81*NrankMax], sdata[(Nthreads+81)*NrankMax]; 
  float x;
  int tid, nt0, tid0, bid, i, nid, Nrank, NT, Nfilt;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  Nfilt    	=   (int) Params[1];
  NT      	=   (int) Params[0];
  Nrank     = (int) Params[6];
  nt0       = (int) Params[9];
  
  if(tid<nt0*((int) Params[6]))
      sW[tid]= W[tid%nt0 + (bid + Nfilt * (tid/nt0))* nt0];
  __syncthreads();
  
  tid0 = 0;
  while (tid0<NT-Nthreads-nt0+1){
	  if (tid<nt0*NrankMax) sdata[tid%nt0 + (tid/nt0)*(Nthreads+nt0)] = 
			data[tid0 + tid%nt0+ NT*(bid + Nfilt*(tid/nt0))];
	  #pragma unroll 3
      for(nid=0;nid<Nrank;nid++){
          sdata[tid + nt0+nid*(Nthreads+nt0)] = data[nt0+tid0 + tid+ NT*(bid +nid*Nfilt)];
	  }
	  __syncthreads();
      
	  x = 0.0f;
      for(nid=0;nid<Nrank;nid++){
		  #pragma unroll 4
          for(i=0;i<nt0;i++)
              x    += sW[i + nid*nt0] * sdata[i+tid + nid*(Nthreads+nt0)];
	  }
      conv_sig[tid0  + tid + NT*bid]   = x;
      
      tid0+=Nthreads;
      __syncthreads();
  }
}
///////////////////////////////////////////////////////////////////////////
__global__ void  bestFilter(const double *Params, const float *data, 
        const float *mu, const float *lam, const float *nu, float *xbest, float *err, int *ftype){

  int tid, tid0, i, bid, NT, Nfilt, ibest = 0;
  float Th,  Cf, Ci, xb, Cbest = 0.0f, epu, cdiff;
 
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT 		= (int) Params[0];
  Nfilt 	= (int) Params[1];
  Th 		= (float) Params[2];
  epu       = (float) Params[8];
  
  tid0 = tid + bid * Nthreads;
  if (tid0<NT-1 & tid0>0){
      for (i=0; i<Nfilt;i++){
          Ci = data[tid0 + NT * i] + mu[i] * lam[i];
          Cf = Ci * Ci / (lam[i] + 1.0f) - lam[i]*mu[i]*mu[i];
          
          // add the shift component
          cdiff = data[tid0+1 + NT * i] - data[tid0-1 + NT * i];
          Cf = Cf + cdiff * cdiff / (epu + nu[i]);
          if (Cf > Cbest){
              Cbest 	= Cf;
              xb      = Ci  - mu[i] * lam[i]; /// (lam[i] + 1);
              ibest 	= i;
          }
      }
      if (Cbest > Th*Th){
          err[tid0] 	= Cbest;
          xbest[tid0] 	= xb;
          ftype[tid0] 	= ibest;
      }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	cleanup_spikes(const double *Params, const float *xbest,
        const float *err, const int *ftype, const bool *UtU, int *st, int *id, float *x,
        float *C, int *counter, float *nsp){
  int lockout, curr_token, indx, maxFR, Nfilt, NTOT, tid, bid, NT, tid0,  j;
  volatile __shared__ float sdata[Nthreads+2*81+1];
  volatile __shared__ int id_sh[Nthreads+2*81+1];
  bool flag=0;
  float err0;
 
  lockout   = (int) Params[9] - 1;
  tid 		= threadIdx.x; 
  bid 		= blockIdx.x;
  
  NT      	= (int) Params[0];
  Nfilt 	= (int) Params[1];
  maxFR 	= (int) Params[3];
  tid0 		= bid * Nthreads;

  if(tid0<NT-Nthreads-2*lockout-1){       
    if (tid<2*lockout){
		sdata[tid] = abs(err[tid0 + tid]*err[tid0 + tid]);
		id_sh[tid] = ftype[tid0 + tid];
	 }
    sdata[tid+2*lockout] = abs(err[2*lockout + tid0 + tid]*err[2*lockout + tid0 + tid]);
	id_sh[tid+2*lockout] = ftype[2*lockout + tid0 + tid];
	
    __syncthreads();
    
    err0 = sdata[tid+lockout];
    curr_token = id_sh[tid+lockout];
    if(err0>1e-10){
        flag = 0;
        for(j=-lockout;j<=lockout;j++)
            if(sdata[tid+lockout+j]>err0)
                if (UtU[curr_token*Nfilt + id_sh[tid+lockout+j]]){
                    flag = 1;
                    break;
                }
        if(flag==0){
            indx = atomicAdd(&counter[0], 1);
            if (indx<maxFR){
                st[indx] = tid+lockout         + tid0;
                id[indx] = ftype[tid+lockout   + tid0];
                x[indx]  = xbest[tid+lockout     + tid0];
                C[indx]  = err0;
                //           atomicAdd(&muout[ftype[tid+lockout   + tid0]], xbest[tid+lockout     + tid0]);
                atomicAdd(&nsp[ftype[tid+lockout   + tid0]], 1);
            }
        }
    }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void average_snips(const double *Params, const int *st, const int *id,
        const float *x,  const int *counter, const float *dataraw, float *WU){
  int nt0, tidx, tidy, bid, i, ind, NT, Nchan;
  float xsum = 0.0f, pm; 
  Nchan = (int) Params[5];
  
  pm = (float) Params[7];

  NT = (int) Params[0];  
  nt0       = (int) Params[9];
  
  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;
  bid 		= blockIdx.x;
  
  for(ind=0; ind<counter[0];ind++)
      if (id[ind]==bid){
		  tidy 		= threadIdx.y;
		  while (tidy<Nchan){	
			xsum = dataraw[st[ind]+tidx + NT * tidy]; 
			WU[tidx+tidy*nt0 + nt0*Nchan * bid] = 
                    pm*WU[tidx+tidy*nt0 + nt0*Nchan * bid] + (1-pm) * xsum;
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
    /* Declare input variables*/
  double *Params, *d_Params;
  int blocksPerGrid, NT, maxFR, Nchan, nt0;
  int const threadsPerBlock = 1024;

  /* Initialize the MathWorks GPU API. */
  mxInitGPU();

  /* read Params and copy to GPU */
  Params        = (double*) mxGetData(prhs[0]);
  NT            = (int) Params[0];
  blocksPerGrid	= (int) Params[1];
  maxFR         = (int) Params[3];
  Nchan         = (int) Params[5];
  nt0           = (int) Params[9];
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);
  
  /* collect input GPU variables*/
  mxGPUArray const  *W, *dataraw,   *data, *UtU, *mu, *lam, *nu;
  const float      *d_W, *d_dataraw, *d_data, *d_mu, *d_lam, *d_nu;
  float *d_dWU;
  const bool *d_UtU;
  
  mxGPUArray *dWU;
  dWU       = mxGPUCopyFromMxArray(prhs[7]);
  d_dWU     = (float *)(mxGPUGetData(dWU));
  
  dataraw       = mxGPUCreateFromMxArray(prhs[1]);
  d_dataraw     = (float const *)(mxGPUGetDataReadOnly(dataraw));
  W             = mxGPUCreateFromMxArray(prhs[2]);
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  data        	= mxGPUCreateFromMxArray(prhs[3]);
  d_data        = (float const *)(mxGPUGetDataReadOnly(data));
  UtU       	= mxGPUCreateFromMxArray(prhs[4]);
  d_UtU     	= (bool const *)(mxGPUGetDataReadOnly(UtU));
  mu            = mxGPUCreateFromMxArray(prhs[5]);
  d_mu          = (float const *)(mxGPUGetDataReadOnly(mu));
  lam       	= mxGPUCreateFromMxArray(prhs[6]);
  d_lam     	= (float const *)(mxGPUGetDataReadOnly(lam));
  nu            = mxGPUCreateFromMxArray(prhs[8]);
  d_nu          = (float const *)(mxGPUGetDataReadOnly(nu));
  /* allocate new GPU variables*/
  float *d_err, *d_x, *d_dout, *d_C,*d_xbest;
  int *d_st, *d_ftype,  *d_id, *d_counter;
  
  cudaMalloc(&d_dout,   NT * blocksPerGrid* sizeof(float));

  cudaMalloc(&d_err,   NT * sizeof(float));
  cudaMalloc(&d_xbest,   NT * sizeof(float));
  cudaMalloc(&d_ftype, NT * sizeof(int));
  cudaMalloc(&d_st,    maxFR * sizeof(int));
  cudaMalloc(&d_id,    maxFR * sizeof(int));
  cudaMalloc(&d_x,     maxFR * sizeof(float));
  cudaMalloc(&d_counter,   2*sizeof(int));
   cudaMalloc(&d_C,     maxFR * sizeof(float));

  cudaMemset(d_dout,    0, NT * blocksPerGrid * sizeof(float));
  cudaMemset(d_counter, 0, 2*sizeof(int));
  cudaMemset(d_st,      0, maxFR *   sizeof(int));
  cudaMemset(d_id,      0, maxFR *   sizeof(int));
  cudaMemset(d_x,       0, maxFR *    sizeof(float));
  cudaMemset(d_C,       0, maxFR *    sizeof(float));

  mxGPUArray *nsp;
  float  *d_nsp;

  const mwSize dimsmu[] = {blocksPerGrid, 1}; 
  nsp	 		= mxGPUCreateGPUArray(2, dimsmu, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
  d_nsp	 	= (float *)(mxGPUGetData(nsp));
  cudaMemset(d_nsp,       0, blocksPerGrid *    sizeof(float));
  
  //const mwSize dimsdWU[] = {nt0,Nchan,blocksPerGrid}; 
  //dWU 		= mxGPUCreateGPUArray(3, dimsdWU, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);  
  //d_dWU     = (float *)(mxGPUGetData(dWU));
  //cudaMemset(d_dWU, 0,  nt0*Nchan*blocksPerGrid * sizeof(float));

  int *counter;
  counter = (int*) calloc(1,sizeof(int));
  cudaMemset(d_err,     0, NT * sizeof(float));
  cudaMemset(d_ftype,   0, NT * sizeof(int));
  
  Conv1D<<<blocksPerGrid,threadsPerBlock>>>(d_Params, d_data, d_W, d_dout);
  bestFilter<<<NT/Nthreads,threadsPerBlock>>>(d_Params, d_dout, d_mu, d_lam, d_nu,
          d_xbest, d_err, d_ftype);
  cleanup_spikes<<<NT/Nthreads,threadsPerBlock>>>(d_Params,  d_xbest, d_err, 
          d_ftype, d_UtU, d_st, d_id, d_x,  d_C, d_counter, d_nsp);
  
  dim3 block(nt0, 1024/nt0);
  average_snips<<<blocksPerGrid,block>>>(  d_Params, d_st, d_id, d_x, d_counter, d_dataraw, d_dWU);

  cudaMemcpy(counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
 
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(dWU);

  float    *x, *C;
  int *st, *id;
  int minSize;
  if (counter[0]<maxFR)  minSize = counter[0];
  else                   minSize = maxFR;
  const mwSize dimst[] 	= {minSize,1}; 
  plhs[1] = mxCreateNumericArray(2, dimst, mxINT32_CLASS, mxREAL);
  st = (int*) mxGetData(plhs[1]);
  plhs[2] = mxCreateNumericArray(2, dimst, mxINT32_CLASS, mxREAL);
  id = (int*) mxGetData(plhs[2]);
  plhs[3] = mxCreateNumericArray(2, dimst, mxSINGLE_CLASS, mxREAL);
  x =  (float*) mxGetData(plhs[3]);
  plhs[4] = mxCreateNumericArray(2, dimst, mxSINGLE_CLASS, mxREAL);
  C =  (float*) mxGetData(plhs[4]);
  cudaMemcpy(st, d_st, minSize * sizeof(int),   cudaMemcpyDeviceToHost);
  cudaMemcpy(id, d_id, minSize * sizeof(int),   cudaMemcpyDeviceToHost);
  cudaMemcpy(x,   d_x, minSize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(C,   d_C, minSize * sizeof(float), cudaMemcpyDeviceToHost);
  
  plhs[5] 	= mxGPUCreateMxArrayOnGPU(nsp);
  
  cudaFree(d_ftype);
  cudaFree(d_err);
  cudaFree(d_xbest);
  cudaFree(d_st);
  cudaFree(d_id);
  cudaFree(d_x);
  cudaFree(d_C);
  cudaFree(d_counter);
  cudaFree(d_Params);

  cudaFree(d_dout);

  mxGPUDestroyGPUArray(data);
  mxGPUDestroyGPUArray(dataraw);
  mxGPUDestroyGPUArray(dWU);
  mxGPUDestroyGPUArray(UtU);
  mxGPUDestroyGPUArray(mu);
  mxGPUDestroyGPUArray(W);
  mxGPUDestroyGPUArray(lam);
  mxGPUDestroyGPUArray(nsp);  
  
}
