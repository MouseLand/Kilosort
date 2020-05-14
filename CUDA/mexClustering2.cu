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
__global__ void computeCost(const double *Params, const float *uproj, const float *mu, const float *W, 
        const bool *match, const int *iC, const int *call, float *cmax){
    
  int NrankPC,j, NchanNear, tid, bid, Nspikes, Nthreads, k, my_chan, this_chan, Nchan;
  float xsum = 0.0f, Ci, lam; 
  
  Nspikes               = (int) Params[0];
  NrankPC             = (int) Params[1];  
  Nthreads              = blockDim.x;
  lam                   = (float) Params[5];
  NchanNear             = (int) Params[6];
  Nchan                 = (int) Params[7];
    
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  while(tid<Nspikes){
      my_chan = call[tid];
      if (match[my_chan + bid * Nchan]){
          xsum = 0.0f;
          for (k=0;k<NchanNear;k++)
              for(j=0;j<NrankPC;j++){
                  this_chan = iC[k + my_chan * NchanNear];
                    xsum += uproj[j + NrankPC * k + NrankPC*NchanNear * tid] * 
                            W[j + NrankPC * this_chan +  NrankPC*Nchan * bid];                    
              }          
          Ci = max(0.0f, xsum) + lam/mu[bid];
          
          cmax[tid + bid*Nspikes] = Ci * Ci / (1.0f + lam/(mu[bid] * mu[bid])) - lam;          
      }
      tid+= Nthreads;
  }  
}


//////////////////////////////////////////////////////////////////////////////////////////
__global__ void bestFilter(const double *Params,  const bool *match, 
        const int *iC, const int *call, const float *cmax, int *id, float *cx){
    
  int Nchan, tid,tind,bid, ind, Nspikes, Nfilters, Nthreads, Nblocks, my_chan;
  float max_running = 0.0f; 
  
  Nspikes               = (int) Params[0];
  Nfilters              = (int) Params[2];
  Nthreads              = blockDim.x;
  Nblocks               = gridDim.x;
  Nchan                = (int) Params[7];

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  tind = tid + bid * Nthreads;
  
  while (tind<Nspikes){      
      max_running = 0.0f;
      id[tind] = 0;
      my_chan = call[tind];
      
      for(ind=0; ind<Nfilters; ind++)          
          if (match[my_chan + ind * Nchan])
              if (cmax[tind + ind*Nspikes] > max_running){
                  id[tind] = ind;
                  max_running = cmax[tind + ind*Nspikes];
              }
      
              
      cx[tind] = max_running; 
      
      tind += Nblocks*Nthreads; 
  }  
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void average_snips(const double *Params, const int *iC, const int *call, 
        const int *id, const float *uproj, const float *cmax, float *WU){
  
  //Nfilt blocks
  //Thread grid = (NrankPC, NchanNear)
  //This implementation does not work correctly for real data!
  //Since this_chan is function of the spike -- spikes assigned to a given template
  //will have max channels that span a 2-3 channel range -- different (tidx, tidy)
  //pairs can wind up trying to add to the same element of dWU, resulting in 
  //collisions and incorrect results. Use the single-threaded version
  //average_snips_v2 instead. Speed hit is only ~ 5-6 seconds out of 360 sec for a 
  //typical 2 hour Neuropixels 1.0 dataset.
  int my_chan, this_chan, tidx, tidy, bid, ind, Nspikes, NrankPC, NchanNear, Nchan;
  float xsum = 0.0f; 
  
  Nspikes               = (int) Params[0];
  NrankPC             = (int) Params[1];  
  Nchan                = (int) Params[7];
  NchanNear             = (int) Params[6];
    
  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;
  bid 		= blockIdx.x;
  
  for(ind=0; ind<Nspikes;ind++) {
      if (id[ind]==bid){          
          my_chan = call[ind];          
          this_chan = iC[tidy + NchanNear * my_chan];
          xsum = uproj[tidx + NrankPC*tidy +  NrankPC*NchanNear * ind];
          WU[tidx + NrankPC*this_chan + NrankPC*Nchan * bid] +=  xsum;             
      }  
  }

  }

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void average_snips_v2(const double *Params, const int *iC, const int *call, 
        const int *id, const float *uproj, const float *cmax, float *WU){
  
            
  // jic, version with no threading over features, to avoid 
  // collisions when summing WU 
  // run
          
  int my_chan, this_chan, bid, ind, Nspikes, NrankPC, NchanNear, Nchan;
  float xsum = 0.0f; 
  int chanIndex, pcIndex;
  
  Nspikes               = (int) Params[0];
  NrankPC             = (int) Params[1];  
  Nchan                = (int) Params[7];
  NchanNear             = (int) Params[6];
    

  bid 		= blockIdx.x;
  
  for(ind=0; ind<Nspikes;ind++)
      if (id[ind]==bid){     
          my_chan = call[ind];  
          for (chanIndex = 0; chanIndex < NchanNear; ++chanIndex) {
            this_chan = iC[chanIndex + NchanNear * my_chan];
            for (pcIndex = 0; pcIndex < NrankPC; ++pcIndex) {
                xsum = uproj[pcIndex + NrankPC*chanIndex +  NrankPC*NchanNear * ind];
                WU[pcIndex + NrankPC*this_chan + NrankPC*Nchan * bid] +=  xsum;
            }
          }
                    
      }  
   }


//////////////////////////////////////////////////////////////////////////////////////////

__global__ void average_snips_v3(const double *Params, const int *ioff, const int *id, const float *uproj, 
        const float *cmax, float *bigArray){
  

  // jic, version to work with Nfeatures threads
  // have made a big array of Nfeature*NfeatW*Nfilters so projections
  // onto each Nfeature can be summed without collisions
  // after running this, need to sum up each set of Nfeature subArrays
  // to calculate the final NfeatW*Nfilters array

  int tid, bid, ind, Nspikes, Nfeatures, NfeatW;
  float xsum = 0.0f;   

  Nspikes               = (int) Params[0];
  Nfeatures             = (int) Params[1];
  NfeatW                = (int) Params[4];

  tid       = threadIdx.x;      //feature index
  bid 		= blockIdx.x;       //filter index



  

  for(ind=0; ind<Nspikes;ind++) {

      if (id[ind]==bid){
            //uproj is Nfeatures x Nspikes
            xsum = uproj[tid + Nfeatures * ind];
            //add this to the Nfeature-th array of NfeatW at the offset for this spike
            bigArray[ioff[ind] + tid + tid*NfeatW + Nfeatures*NfeatW * bid] +=  xsum; 
      }  //end of if block for  match
  }     //end of loop over spikes

}



__global__ void sum_dWU(const double *Params, const float *bigArray, float *WU) {

  int tid,bid, ind, Nfilters, Nthreads, Nfeatures, Nblocks, NfeatW, nWU, nElem;
  float sum = 0.0f; 
  
  Nfeatures             = (int) Params[1];  //NrankPC, number of pcs
  NfeatW                = (int) Params[4];  //Nchan*nPC
  Nfilters              = (int) Params[2];
  Nthreads              = blockDim.x;
  Nblocks               = gridDim.x;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;


  //WU is NfeatW x Nfilters. 

  nWU = NfeatW * Nfilters;
  nElem = Nfeatures*NfeatW; //number of elements in each subArray of bigArray

  //Calculate which element we're addressing  
  int tind = tid + bid * Nthreads;

  int currFilt, currFW, currIndex;
  while (tind < nWU){


      //which filter and element of WU?
      currFilt = floor((double)(tind/NfeatW));
      currFW = tind - currFilt*NfeatW;

      //Sum up the Nfeature elements of bigArray that correspond to this 
      //filter and NfeatW 

      sum = 0.0f;

      for(ind=0; ind<Nfeatures; ind++) {
          //bigArray is Nfilter arrays of Nfeature x NfeatW;
          currIndex = currFilt*nElem + ind*NfeatW + currFW;
          sum += bigArray[ currIndex ];
      }         

      WU[tind] += sum;
      tind += Nblocks*Nthreads; 

   }  

}



//////////////////////////////////////////////////////////////////////////////////////////
__global__ void count_spikes(const double *Params, const int *id, int *nsp, const float *x, float *V){
    
  int tid, tind, bid, ind, Nspikes, Nfilters, NthreadsMe, Nblocks;
  
  Nspikes               = (int) Params[0];
  Nfilters             = (int) Params[2];
  
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NthreadsMe              = blockDim.x;
  Nblocks               = gridDim.x;
  
  tind = tid + NthreadsMe *bid;
  
  while (tind<Nfilters){
      for(ind=0; ind<Nspikes;ind++)
          if (id[ind]==tind){
              nsp[tind] ++;
              V[tind] += x[tind];
          }
      V[tind] = V[tind] / (.001f + (float) nsp[tind]);
      
      tind += NthreadsMe * Nblocks;
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
  unsigned int Nchan, NrankPC, Nspikes, Nfilters;

  
  /* Initialize the MathWorks GPU API. */
  mxInitGPU();

  /* read Params and copy to GPU */
  Params                = (double*) mxGetData(prhs[0]);
  Nspikes               = (unsigned int) Params[0];  
  Nfilters              = (unsigned int) Params[2];  
  NrankPC             = (unsigned int) Params[1];  
  Nchan                 = (unsigned int) Params[7];
  
  // copy Params to GPU
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);
  
  /* collect input GPU variables*/
  mxGPUArray const  *W, *uproj, *call, *iMatch, *mu, *iC;
  const float *d_W, *d_uproj, *d_mu;
  const int *d_call, *d_iC;  
  float *d_dWU;
  const bool *d_iMatch;
    
  // these come as const GPU Arrays, just transfer them over
  uproj         = mxGPUCreateFromMxArray(prhs[1]);
  W             = mxGPUCreateFromMxArray(prhs[2]);
  mu            = mxGPUCreateFromMxArray(prhs[3]);  
  call          = mxGPUCreateFromMxArray(prhs[4]);  
  iMatch         = mxGPUCreateFromMxArray(prhs[5]);
  iC            = mxGPUCreateFromMxArray(prhs[6]);  

  d_uproj       = (float const *)(mxGPUGetDataReadOnly(uproj));
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  d_mu          = (float const *)  (mxGPUGetDataReadOnly(mu));
  d_call        = (int const *)  (mxGPUGetDataReadOnly(call));
  d_iC          = (int const *)  (mxGPUGetDataReadOnly(iC));    
  d_iMatch      = (bool const *)  (mxGPUGetDataReadOnly(iMatch));  

  mxGPUArray *dWU;
  const mwSize ddWU[] 	= {NrankPC * Nchan, Nfilters};
  dWU 		= mxGPUCreateGPUArray(2, ddWU, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  d_dWU 		= (float *)(mxGPUGetData(dWU));
  
  /* Define new GPU variables*/
  float *d_cmax,  *d_x, *d_V;
  int *d_id, *d_nsp;
  
  // allocate a lot of GPU variables
  cudaMalloc(&d_cmax,    Nspikes * Nfilters *  sizeof(float));
  cudaMalloc(&d_id,      Nspikes  *  sizeof(int));
  cudaMalloc(&d_x,      Nspikes  *  sizeof(float));
  cudaMalloc(&d_nsp,      Nfilters  *  sizeof(int));
     
  cudaMemset(d_nsp,      0, Nfilters *   sizeof(int));
  cudaMemset(d_dWU, 0, NrankPC*Nchan*Nfilters  *  sizeof(float));
  
  //jic add Memset for d_V
  cudaMalloc(&d_V,      Nfilters  *  sizeof(float));
  cudaMemset(d_V, 0, Nfilters  *  sizeof(float));
  
  
  // get list of cmaxes for each combination of neuron and filter
  computeCost<<<Nfilters, 1024>>>(d_Params, d_uproj, d_mu, d_W, 
          d_iMatch,d_iC, d_call, d_cmax);

  // loop through cmax to find best template
  bestFilter<<<40, 256>>>(d_Params, d_iMatch, d_iC, d_call, d_cmax, d_id, d_x);
  
  // average all spikes for same template -- ORIGINAL
//   dim3 thNN(NrankPC, NchanNear);
//   average_snips<<<Nfilters, thNN>>>(d_Params, d_iC, d_call, d_id, d_uproj, d_cmax, d_dWU);

  // average all spikes for same template -- threaded over filters, but not features
  // avoid collision when adding to elements of d_dWU
  average_snips_v2<<<Nfilters, 1>>>(d_Params, d_iC, d_call, d_id, d_uproj, d_cmax, d_dWU);

  //-------------------------------------------------
  //jic for running average_snips_v3 with Nfeature threads  
//   float *d_bigArray;
//   int bSize;
//   int NfeatW = (int) Params[4];
//   int Nfeatures = (int) Params[1];
//   bSize = Nfeatures*NfeatW*Nfilters;
//   cudaMalloc(&d_bigArray, bSize*sizeof(float) );
//   cudaMemset(d_bigArray, 0, bSize*sizeof(float) );
// 
//   average_snips_v3<<<Nfilters, Nfeatures>>>(d_Params, d_ioff, d_id,  
//       d_uproj, d_cmax, d_bigArray);
//   sum_dWU<<<128,1024>>>( d_Params, d_bigArray, d_dWU );
//   cudaFree(d_bigArray);
 //-------------------------------------------------


  count_spikes<<<7, 256>>>(d_Params, d_id, d_nsp, d_x, d_V);

  // dWU stays a GPU array
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(dWU);
  
  // put these ones on the CPU side: id, cmax, cf, nsp 
  int *id, *nsp;
  float *x, *V;
  
  const mwSize dimst[]      = {Nspikes,1};  
  const mwSize dimst2[] 	= {Nspikes,Nfilters};  
  const mwSize dimst4[] 	= {Nfilters,1};  

  plhs[1]   = mxCreateNumericArray(2, dimst,  mxINT32_CLASS,  mxREAL);
  plhs[2]   = mxCreateNumericArray(2, dimst, mxSINGLE_CLASS, mxREAL);  
  plhs[3]   = mxCreateNumericArray(2, dimst4, mxINT32_CLASS,  mxREAL);  
  plhs[4]   = mxCreateNumericArray(2, dimst2, mxSINGLE_CLASS, mxREAL);  

  id        = (int*) mxGetData(plhs[1]);  
  x        = (float*) mxGetData(plhs[2]);  
  nsp       = (int*) mxGetData(plhs[3]);  
  V        = (float*) mxGetData(plhs[4]);  
  
  cudaMemcpy(id,   d_id,  Nspikes * sizeof(int),   cudaMemcpyDeviceToHost);
  cudaMemcpy(x, d_x,Nspikes * sizeof(float),  cudaMemcpyDeviceToHost);
  cudaMemcpy(nsp,  d_nsp, Nfilters * sizeof(int),   cudaMemcpyDeviceToHost);
  cudaMemcpy(V, d_cmax, Nspikes * Nfilters  * sizeof(float),  cudaMemcpyDeviceToHost);
  
  //we are done, clear everything from the GPU
  cudaFree(d_Params);
  cudaFree(d_cmax);
  cudaFree(d_x);
  cudaFree(d_V);
  cudaFree(d_id);
  cudaFree(d_nsp);
  

  //do this for the constant variables
  mxGPUDestroyGPUArray(uproj);
  mxGPUDestroyGPUArray(dWU);  
  mxGPUDestroyGPUArray(W);    
  mxGPUDestroyGPUArray(iC);  
  mxGPUDestroyGPUArray(iMatch);  
  mxGPUDestroyGPUArray(call);  
  mxGPUDestroyGPUArray(mu);  

  
}
