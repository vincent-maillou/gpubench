#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include <assert.h>

#include <cuda_runtime.h>

using namespace std;


inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}



/* -----------------------------------------------------
                 Kernel Definitions
----------------------------------------------------- */ 

__global__
void initVector(double *X, long N)
    {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = index; i < N; i += stride)
      {
      X[i] = cos(2.*M_PI*i/N);
      }
    }



int main(int argc,char**argv)
{
  /* -----------------------------------------------------
                  Parsing des arguments
  ----------------------------------------------------- */ 

  for(int i=0;i<argc;i++)
    cout << argv[i] << " " ;
  cout << endl;

  bool lvlOut = false;
  int  A = 1;
  int  M = 1;
  long N = 100000; 

  for(int i=0;i<argc;i++) // Parsing des options d'execution
  {
    if(argv[i]==string("-n"))
      N = stol(argv[i+1]);
    if(argv[i]==string("-m"))
      M = stol(argv[i+1]);
    if(argv[i]==string("-a"))
      A = stol(argv[i+1]);
    if(argv[i]==string("-out"))
      lvlOut = true;
  } 

  printf("Vector initialzation in parallel using CUDA:\n");



  /* -----------------------------------------------------
            Initialisation paramètres CUDA
  ----------------------------------------------------- */ 

  int deviceId;
  int numberOfSMs;

  checkCuda(cudaGetDevice(&deviceId));
  checkCuda(cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

  size_t threadsPerBlock = 128;
  size_t numberOfBlocks = 32 * numberOfSMs;

  // Allocation et initialisation des vecteurs sur le GPU
  double *X_gpu;
  
  cudaMalloc(&X_gpu, sizeof(double)*N);
   
  auto begin = std::chrono::high_resolution_clock::now(); 
  
  for(int k=0;k<A;k++)
    {
    for(int i=0;i<M;i++) // Lunching 'M' kernels
      {
      initVector<<<numberOfBlocks, threadsPerBlock>>>(X_gpu, N);
      cudaDeviceSynchronize();
      }
    }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  
  cout << "   Vector initialization in " << elapsed.count()/(1000.0*A) << " us" << endl;

  if(lvlOut)
    {
    double *X_host;

    cudaMallocHost(&X_host, sizeof(double)*N);

    cudaMemcpy(X_host, X_gpu, sizeof(double)*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cout << "   Vecteur X: ";
    for(int i=0;i<N;i++)
      {
      cout << X_host[i] << " ";
      }
    cout << endl;
    cudaFree(X_host);
    }

  cout << "   Completed for: " << N << " size, " << M << " repetition and averaged " << A << " times" << endl;
  
  cudaFree(X_gpu);
  
  return 0 ;
}
