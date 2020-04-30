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

      tid0 = tid0 + blockDim.x * gridDim.x;
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
