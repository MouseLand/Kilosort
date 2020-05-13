const int nblock = 32;
//////////////////////////////////////////////////////////////////////////////////////////

__global__ void	crossFilter(const double *Params, const float *W1, const float *W2,
        const float *UtU, float *WtW){
  __shared__ float shW1[nblock*81], shW2[nblock*81];

  float x;
  int nt0, tidx, tidy , bidx, bidy, i, Nfilt, t, tid1, tid2;

  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;
  bidx 		= blockIdx.x;
  bidy 		= blockIdx.y;

  Nfilt     = (int) Params[1];
  nt0       = (int) Params[9];

  tid1 = tidx + bidx*nblock;

  tid2 = tidy + bidx*nblock;
  if (tid2<Nfilt){
      while(tidx<nt0){
          shW1[tidx + tidy * nt0] = W1[tidx + tid2 * nt0];
          tidx+= nblock;
      }
  }
  tidx 		= threadIdx.x;
  tid2      = tidy + bidy*nblock;
  if (tid2<Nfilt){
      while(tidx<nt0){
          shW2[tidx + tidy * nt0] = W2[tidx + tid2 * nt0];
          tidx+= nblock;
      }
  }
  tidx 		= threadIdx.x;

  __syncthreads();

  if (tid2<Nfilt && tid1<Nfilt){
      for(i=0;i<2*nt0-1;i++){
          x = 0.0f;
          if(i<nt0)
              for(t=0;t<i+1;t++)
                  x += shW1[t + nt0 * tidx] * shW2[t + (nt0-i-1) + nt0 * tidy];
          else
              for(t=i-nt0+1;t<nt0;t++)
                  x += shW1[t + nt0 * tidx] * shW2[t + (nt0-i-1) + nt0 * tidy];

          WtW[tid1 + tid2*Nfilt +  i*Nfilt*Nfilt] =
                  x * UtU[tid1 + tid2*Nfilt];
      }
  }
}
