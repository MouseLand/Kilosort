__global__ void computeCost(const double *Params, const float *Ws, const float *mus,
        const float *W, const float *mu, const bool *iMatch,
        const int *iC, const int *Wh, float *cmax){

  int j, tid, bid, Nspikes, my_chan, this_chan, Nchan, NrankPC, NchanNear, Nthreads, k;
  float xsum = 0.0f, Ci;

  Nspikes               = (int) Params[0];
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
