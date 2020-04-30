const int  Nthreads = 1024, maxFR = 10000, NrankMax = 3, nt0max=81, NchanMax = 17;

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

    float x;
    int tidx, nt0min, tidy, my_chan, this_chan, tid, bid, nt0, NchanNear, j, t, NT, NrankPC;
    volatile __shared__ float sW[nt0max*NrankMax], sD[nt0max*NchanMax];

    NT 		= (int) Params[0];
    NchanNear = (int) Params[2];
    nt0       = (int) Params[3];
    NrankPC  = (int) Params[6];
    nt0min    = (int) Params[4];

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
__global__ void maxChannels(const double *Params, const float *dataraw, const float *data,
	const int *iC, int *st, int *id, int *counter){

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
