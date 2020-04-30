const int  Nthreads = 1024,  NrankMax = 3, nt0max = 71, NchanMax = 1024;

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void blankdWU(const double *Params, const double *dWU,
        const int *iC, const int *iW, double *dWUblank){

  int nt0, tidx, tidy, bid, Nchan, NchanNear, iChan;

  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
  NchanNear = (int) Params[10];

  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;

  bid 		= blockIdx.x;

  while (tidy<NchanNear){
      iChan = iC[tidy+ NchanNear * iW[bid]];
      dWUblank[tidx + nt0*iChan + bid * nt0 * Nchan] =
              dWU[tidx + nt0*iChan + bid * nt0 * Nchan];
      tidy+=blockDim.y;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void getwtw(const double *Params, const double *dWU, double *wtw){

  int nt0, tidx, tidy, bid, Nchan,k;
  double x;

  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];

  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;

  bid 		= blockIdx.x;

  while (tidy<nt0){
      x = 0.0f;
      for (k=0; k<Nchan; k++)
          x += dWU[tidx + k*nt0 + bid * Nchan*nt0] *
                  dWU[tidy + k*nt0 + bid * Nchan*nt0];
      wtw[tidx + tidy*nt0 + bid * nt0*nt0] = x;

      tidy+=blockDim.y;
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void getU(const double *Params, const double *dWU, double *W, double *U){

  int Nfilt, nt0, tidx, tidy, bid, Nchan,k;
  double x;


  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
  Nfilt    	=   (int) Params[1];
  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;
  bid 		= blockIdx.x;

  while (tidy<Nchan){
      x = 0.0f;
      for (k=0; k<nt0; k++)
          x += W[k + nt0*bid + nt0*Nfilt*tidx] *
                  dWU[k + tidy*nt0 + bid * Nchan*nt0];
      U[tidy + Nchan * bid + Nchan * Nfilt * tidx] = x;

      tidy+=blockDim.y;
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void getW(const double *Params, double *wtw, double *W){

  int Nfilt, nt0, tid, bid, i, t, Nrank,k, tmax;
  double x, x0, xmax;
  volatile __shared__ double sW[nt0max*NrankMax], swtw[nt0max*nt0max], xN[1];

  nt0       = (int) Params[4];
   Nrank       = (int) Params[6];
  Nfilt    	=   (int) Params[1];
  tmax = (int) Params[11];

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;

  for (k=0;k<nt0;k++)
      swtw[tid + k*nt0] = wtw[tid + k*nt0 + bid * nt0 * nt0];
  for (k=0;k<Nrank;k++)
      sW[tid + k*nt0] = W[tid + bid * nt0  + k * nt0*Nfilt];
  __syncthreads();


  // for each svd
  for(k=0;k<Nrank;k++){
      for (i=0;i<100;i++){
          // compute projection of wtw
          x = 0.0f;
          for (t=0;t<nt0;t++)
              x+= swtw[tid + t*nt0] * sW[t + k*nt0];

          __syncthreads();
          if (i<99){
              sW[tid + k*nt0] = x;
              __syncthreads();

              if (tid==0){
                  x0 = 0.00001f;
                  for(t=0;t<nt0;t++)
                      x0+= sW[t + k*nt0] * sW[t + k*nt0];
                  xN[0] = sqrt(x0);
              }
              __syncthreads();

              sW[tid + k*nt0] = x/xN[0];
              __syncthreads();
          }
      }

      // now subtract off this svd from wtw
      for (t=0;t<nt0;t++)
          swtw[tid + t*nt0] -= sW[t+k*nt0] * x;

      __syncthreads();
  }


  xmax = sW[tmax];
  __syncthreads();

  sW[tid] = - sW[tid] * copysign(1.0, xmax);

  // now write W back
  for (k=0;k<Nrank;k++)
      W[tid + bid * nt0  + k * nt0*Nfilt] = sW[tid + k*nt0];

}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void reNormalize(const double *Params, const double *A, const double *B,
        double *W, double *U, double *mu){

    int Nfilt, nt0, tid, bid, Nchan,k, Nrank, imax, t, ishift, tmax;
    double x, xmax, xshift, sgnmax;

    volatile __shared__ double sW[NrankMax*nt0max], sU[NchanMax*NrankMax], sS[NrankMax+1],
            sWup[nt0max*10];

    nt0       = (int) Params[4];
    Nchan     = (int) Params[9];
    Nfilt     = (int) Params[1];
    Nrank     = (int) Params[6];
    tmax = (int) Params[11];
    bid 	  = blockIdx.x;

    tid 		= threadIdx.x;
    for(k=0;k<Nrank;k++)
        sW[tid + k*nt0] = W[tid + bid*nt0 + k*Nfilt*nt0];

    while (tid<Nchan*Nrank){
        sU[tid] = U[tid%Nchan + bid*Nchan  + (tid/Nchan)*Nchan*Nfilt];
        tid += blockDim.x;
    }

    __syncthreads();

    tid 		= threadIdx.x;
    if (tid<Nrank){
        x = 0.0f;
        for (k=0; k<Nchan; k++)
            x += sU[k + tid*Nchan] * sU[k + tid*Nchan];
        sS[tid] = sqrt(x);
    }
    // no need to sync here
    if (tid==0){
        x = 0.0000001f;
        for (k=0;k<Nrank;k++)
            x += sS[k] * sS[k];
        sS[Nrank] = sqrt(x);
        mu[bid] = sqrt(x);
    }

    __syncthreads();

    // now re-normalize U
    tid 		= threadIdx.x;

    while (tid<Nchan*Nrank){
        U[tid%Nchan + bid*Nchan  + (tid/Nchan)*Nchan*Nfilt] = sU[tid] / sS[Nrank];
        tid += blockDim.x;
    }

    /////////////
    __syncthreads();

    // now align W
    xmax = 0.0f;
    imax = 0;
    for(t=0;t<nt0;t++)
        if (abs(sW[t]) > xmax){
            xmax = abs(sW[t]);
            imax = t;
        }

    tid 		= threadIdx.x;
    // shift by imax - tmax
    for (k=0;k<Nrank;k++){
        ishift = tid + (imax-tmax);
        ishift = (ishift%nt0 + nt0)%nt0;

        xshift = sW[ishift + k*nt0];
        W[tid + bid*nt0 + k*nt0*Nfilt] = xshift;
    }

    __syncthreads();
     for (k=0;k<Nrank;k++){
        sW[tid + k*nt0] = W[tid + bid*nt0 + k*nt0*Nfilt];
    }

    /////////////
    __syncthreads();

        // now align W. first compute 10x subsample peak
    tid 		= threadIdx.x;
    if (tid<10){
        sWup[tid] = 0;
        for (t=0;t<nt0;t++)
            sWup[tid] += A[tid + t*10] * sW[t];
    }
    __syncthreads();

    xmax = 0.0f;
    imax = 0;
    sgnmax = 1.0f;
    for(t=0;t<10;t++)
        if (abs(sWup[t]) > xmax){
            xmax = abs(sWup[t]);
            imax = t;
            sgnmax = copysign(1.0f, sWup[t]);
        }

    // interpolate by imax
    for (k=0;k<Nrank;k++){
        xshift = 0.0f;
        for (t=0;t<nt0;t++)
            xshift += B[tid + t*nt0 +nt0*nt0*imax] * sW[t + k*nt0];

        if (k==0)
            xshift = -xshift * sgnmax;

        W[tid + bid*nt0 + k*nt0*Nfilt] = xshift;
    }

}
