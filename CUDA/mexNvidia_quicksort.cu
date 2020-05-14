
/* adapted from NVIDIA CUDA code samples to return array of indicies along 
/* with the sorted array.
/*
/* host code must include this file (#include "mexNvidia_quicksort.cu") and
/* call "set_idx" to create the array of indices which will track the sort
/*
/* sample host code (commented out) at end of file
/*
* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#include <iostream>
#include <cstdio>
//#include <helper_cuda.h>
//#include <helper_string.h>

#define MAX_DEPTH       16
#define INSERTION_SORT  32

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort( unsigned int *data, unsigned int *idx, int left, int right )
{
  
  unsigned int min_val_idx;

  for( int i = left ; i <= right ; ++i )
  {
    unsigned min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for( int j = i+1 ; j <= right ; ++j )
    {
      unsigned val_j = data[j];
      if( val_j < min_val )
      {
        min_idx = j;
        min_val = val_j;
        min_val_idx = idx[j];
      }
    }

    // Swap the values.
    if( i != min_idx )
    {
      data[min_idx] = data[i];
      data[i] = min_val;
      idx[min_idx] = idx[i];
      idx[i] = min_val_idx;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort( unsigned int *data, unsigned int *idx, int left, int right, int depth )
{
  // If we're too deep or there are few elements left, we use an insertion sort...
  if( depth >= MAX_DEPTH || right-left <= INSERTION_SORT )
  {
    selection_sort( data, idx, left, right );
    return;
  }

  unsigned int *lptr = data+left;
  unsigned int *rptr = data+right;
  unsigned int  pivot = data[(left+right)/2];

  unsigned int *lptr_idx = idx+left;
  unsigned int *rptr_idx = idx+right;

  // Do the partitioning.
  while(lptr <= rptr)
  {
    // Find the next left- and right-hand values to swap
    unsigned int lval = *lptr; 
    unsigned int rval = *rptr;
    unsigned int lval_idx = *lptr_idx;
    unsigned int rval_idx = *rptr_idx;

    // Move the left pointer as long as the pointed element is smaller than the pivot.
    while( lval < pivot )
    {
      lptr++;
      lval = *lptr;
      lptr_idx++;
      lval_idx = *lptr_idx;
    }

    // Move the right pointer as long as the pointed element is larger than the pivot.
    while( rval > pivot )
    {
      rptr--;
      rval = *rptr;
      rptr_idx--;
      rval_idx = *rptr_idx;
    }

    // If the swap points are valid, do the swap!
    if(lptr <= rptr)
    {
      *lptr++ = rval;
      *rptr-- = lval;
      *lptr_idx++ = rval_idx;
      *rptr_idx-- = lval_idx;
    }
  }

  // Now the recursive part
  int nright = rptr - data;
  int nleft  = lptr - data;

  // Launch a new block to sort the left part.
  if(left < (rptr-data)) 
  {
    cudaStream_t s;
    cudaStreamCreateWithFlags( &s, cudaStreamNonBlocking );
    cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, idx, left, nright, depth+1);
    cudaStreamDestroy( s );
  }

  // Launch a new block to sort the right part.
  if((lptr-data) < right) 
  {
    cudaStream_t s1;
    cudaStreamCreateWithFlags( &s1, cudaStreamNonBlocking );
    cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, idx, nleft, right, depth+1);
    cudaStreamDestroy( s1 );
  }
}


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

// will need to include this function to set indicies in the calling program
// to allow sorting to be optional.
// create gpu array of starting index values, 0..nitimes-1
// call with no threads, i.e. <<1, 1>>
// __global__ void set_idx( unsigned int *idx, const unsigned int nitems ) {
//     for( int i = 0; i < nitems; ++ i ) {
//         idx[i] = i;
//     }
// }

// copy values from an integer to a new array in sort order given by sort_idx
// call with no threads, i.e. <<1, 1>>
__global__ void copy_sort_int( const int *orig, const unsigned int *sort_idx, 
        const unsigned int nitems, int *sorted ) {
    for( int i = 0; i < nitems; ++ i ) {
        sorted[sort_idx[i]] = orig[i];
    }
}

// copy values from an array of single precision
// floating point numbers to a new array in sort order given by sort_idx
// call with no threads, i.e. <<1, 1>>
__global__ void copy_sort_int( const float *orig, const unsigned int *sort_idx, 
        const unsigned int nitems, float *sorted ) {
    for( int i = 0; i < nitems; ++ i ) {
        sorted[sort_idx[i]] = orig[i];
    }
}

///////////////////////////////////////////////////////////////////
// Host code
//////////////////////////////////////////////////////////////////
//use for testing the sort, calling from matlab
//reproducing the c code from the nvidia example
//
//void mexFunction(int nlhs, mxArray *plhs[],
//                 int nrhs, mxArray const *prhs[])
//{
//  /* Initialize the MathWorks GPU API. */
//  mxInitGPU();

//  /* Declare input variables*/
//  const unsigned int *d_input;
//  double *Params;
//  int nitems;

//  /* read Params into appropriately typed variables */
//  Params  	= (double*) mxGetData(prhs[0]);
//  nitems    = (int) Params[0];
  
//  mxGPUArray const *input;

  //for a constant GPU array, just need to "create" a GPU equivalent
  //plus a pointer to that array to send to functions (don't understand 
  //why the latter is true, really
//  input   = mxGPUCreateFromMxArray(prhs[1]);
//  d_input = (unsigned int const *)(mxGPUGetDataReadOnly(input));

  //for output variables, declare, allocate space, and initialize
  //these are local gpu arrays

//  unsigned int *d_output, *d_idx;
//  cudaMalloc(&d_output,  nitems * sizeof(int));
//  cudaMemset(d_output, 0, nitems *sizeof(int));

//  cudaMalloc(&d_idx,  nitems * sizeof(int));
//  cudaMemset(d_idx, 0, nitems *sizeof(int));

  //copy input to output; quicksort acts on that array
//  cudaMemcpy( d_output, d_input, nitems*sizeof(int), cudaMemcpyDeviceToDevice );

  //fill the idx array with integers from 0 to nitems-1
//  set_idx<<< 1, 1 >>>(d_idx, nitems);
  

  // Launch quicksort on device
//  int left = 0;
//  int right = nitems-1;
  //cdp_simple_quicksort<<< 1, 1 >>>(d_output, d_idx, left, right, 0);

//  const mwSize dimst[] 	= {nitems,1};
//  unsigned int *output, *idx;

//  plhs[0] = mxCreateNumericArray(2, dimst, mxUINT32_CLASS, mxREAL);
//  output = (unsigned int*) mxGetData(plhs[0]); 
//  cudaMemcpy(output, d_output, nitems * sizeof(int), cudaMemcpyDeviceToHost);

//  plhs[1] = mxCreateNumericArray(2, dimst, mxUINT32_CLASS, mxREAL);
//  idx = (unsigned int*) mxGetData(plhs[1]); 
//  cudaMemcpy(idx, d_idx, nitems * sizeof(int), cudaMemcpyDeviceToHost);

//  cudaFree(d_output);
//  cudaFree(d_idx);
//  mxGPUDestroyGPUArray(input);
//}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
//void run_qsort(unsigned int *data, unsigned int nitems)
//{
  // Prepare CDP for the max depth 'MAX_DEPTH'.
//  checkCudaErrors( cudaDeviceSetLimit( cudaLimitDevRuntimeSyncDepth, MAX_DEPTH ) );

  // Launch on device
//  int left = 0;
//  int right = nitems-1;
//  std::cout << "Launching kernel on the GPU" << std::endl;
//  cdp_simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
//  checkCudaErrors(cudaDeviceSynchronize());
//}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
//void initialize_data(unsigned int *dst, unsigned int nitems)
//{
  // Fixed seed for illustration
//  srand(2047);

  // Fill dst with random values
//  for (unsigned i = 0 ; i < nitems ; i++)
//    dst[i] = rand() % nitems ;
//}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
//void check_results( int n, unsigned int *results_d )
//{
//  unsigned int *results_h = new unsigned[n];
//  checkCudaErrors( cudaMemcpy( results_h, results_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost ));
//  for( int i = 1 ; i < n ; ++i )
//    if( results_h[i-1] > results_h[i] )
//    {
//      std::cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << std::endl;
//      exit(EXIT_FAILURE);
//    }
//  std::cout << "OK" << std::endl;
//  delete[] results_h;
//}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char **argv)
//{
//  int num_items = 128;
//  bool verbose = false;

//  if (checkCmdLineFlag( argc, (const char **)argv, "help" ) ||
//	  checkCmdLineFlag( argc, (const char **)argv, "h" ))
//  {
//      std::cerr << "Usage: " << argv[0] << " num_items=<num_items>\twhere num_items is the number of items to sort" << std::endl;
//      exit(EXIT_SUCCESS);
//  }

//  if (checkCmdLineFlag( argc, (const char **)argv, "v"))
//  {
//      verbose = true;
//  }
//  if (checkCmdLineFlag( argc, (const char **)argv, "num_items"))
//  {
//      num_items = getCmdLineArgumentInt( argc, (const char **)argv, "num_items");
//      if( num_items < 1 )
//      {
//        std::cerr << "ERROR: num_items has to be greater than 1" << std::endl;
//        exit(EXIT_FAILURE);
//      }
//  }

  // Get device properties
//  int device_count = 0, device = -1;
//  checkCudaErrors( cudaGetDeviceCount( &device_count ) );
//  for( int i = 0 ; i < device_count ; ++i )
//  {
//    cudaDeviceProp properties;
//    checkCudaErrors( cudaGetDeviceProperties( &properties, i ) );
//    if( properties.major > 3 || ( properties.major == 3 && properties.minor >= 5 ) )
//    {
//      device = i;
//      std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
//      break;
//   }
//    std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
//  }
//  if( device == -1 )
//  {
//    std::cerr << "cdpSimpleQuicksort requires GPU devices with compute SM 3.5 or higher.  Exiting..." << std::endl;
//    exit(EXIT_SUCCESS);
//  }
//  cudaSetDevice(device);

  // Create input data
// unsigned int *h_data = 0;
//  unsigned int *d_data = 0;

  // Allocate CPU memory and initialize data.
//  std::cout << "Initializing data:" << std::endl;
//  h_data =(unsigned int *)malloc( num_items*sizeof(unsigned int));
//  initialize_data(h_data, num_items);
//  if( verbose )
//  {
//    for(int i=0 ; i<num_items ; i++)
//      std::cout << "Data [" << i << "]: " << h_data[i] << std::endl;
//  }
  
  // Allocate GPU memory.
//  checkCudaErrors(cudaMalloc((void **)&d_data, num_items * sizeof(unsigned int)));
//  checkCudaErrors(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int), cudaMemcpyHostToDevice));

  // Execute
//  std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
//  run_qsort(d_data, num_items);
  
  // Check result
//  std::cout << "Validating results: ";
//  check_results(num_items, d_data);

//  free(h_data);
//  checkCudaErrors( cudaFree(d_data));
// cudaDeviceReset();
//  exit( EXIT_SUCCESS );
//}
