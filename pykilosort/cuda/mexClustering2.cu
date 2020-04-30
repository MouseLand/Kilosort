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

  int my_chan, this_chan, tidx, tidy, bid, ind, Nspikes, NrankPC, NchanNear, Nchan;
  float xsum = 0.0f;

  Nspikes               = (int) Params[0];
  NrankPC             = (int) Params[1];
  Nchan                = (int) Params[7];
  NchanNear             = (int) Params[6];

  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;
  bid 		= blockIdx.x;

  for(ind=0; ind<Nspikes;ind++)
      if (id[ind]==bid){
          my_chan = call[ind];
          this_chan = iC[tidy + NchanNear * my_chan];
          xsum = uproj[tidx + NrankPC*tidy +  NrankPC*NchanNear * ind];
          WU[tidx + NrankPC*this_chan + NrankPC*Nchan * bid] +=  xsum;
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

  Nfeatures             = (int) Params[1];
  NfeatW                = (int) Params[4];
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
