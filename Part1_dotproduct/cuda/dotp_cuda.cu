#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include <assert.h>

#include <cuda_runtime.h>
#include "cublas_v2.h" 

using namespace std ;

#define D_vectAdd     0
#define D_dotProduct  1

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
void initVector(double *X, double *Y, long N)
    {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = index; i < N; i += stride)
      {
      X[i] = cos(2.*M_PI*i/N);
      Y[i] = sin(2.*M_PI*i/N);
      }
    }

__global__
void elementwiseMultVect(double *X, double *Y, long N)
  {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;  

  for(int i = index; i < N; i += stride)
    {
    X[i] = X[i] * Y[i];
    }
  }

__global__
void sum_reduction_kernel_ext(double *X, long N)
  {
  uint index       = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride      = blockDim.x * gridDim.x;  
  long sum_stride  = ceil(N/2.);

  for(int i = index; i < N; i += stride)
    {
      if(i+sum_stride<N)
        X[i] = X[i] + X[i+sum_stride];
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
  long N  = 100000; 

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

  printf("Dot product in parallel using CUDA:\n");



  /* -----------------------------------------------------
            Initialisation paramètres CUDA
  ----------------------------------------------------- */ 

  int deviceId;
  int numberOfSMs;

  checkCuda(cudaGetDevice(&deviceId));
  checkCuda(cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

  size_t threadsPerBlock = 128;
  size_t numberOfBlocks = 32 * numberOfSMs;

  cublasHandle_t h_cuBLAS;
  cublasCreate(&h_cuBLAS);
  cublasSetPointerMode(h_cuBLAS, CUBLAS_POINTER_MODE_DEVICE);

  // Allocation et initialisation des vecteurs sur le GPU
  double *X_gpu, *Y_gpu, *dotp_gpu, *dotp_host;
  
  cudaMalloc(&X_gpu, sizeof(double)*N);
  cudaMalloc(&Y_gpu, sizeof(double)*N);
  cudaMalloc(&dotp_gpu, sizeof(double)); // Stockage du résultat de cuBLAS dot product sur le GPU
  cudaMallocHost(&dotp_host, sizeof(double)); // Resultat du dotp, host side
   

  /* Initialisation des vecteurs sur le gpu */

  initVector<<<numberOfBlocks, threadsPerBlock>>>(X_gpu, Y_gpu, N);
  cudaDeviceSynchronize();



  /* -----------------------------------------------------
                Produit scalaire
  ----------------------------------------------------- */ 
  
  long length_reduction = N;

  auto begin = std::chrono::high_resolution_clock::now(); 
  
  for(int k=0;k<A;k++)
    {
    for(int i=0; i<M; i++) // Lunching 'repeat_trial' kernels
      {
      elementwiseMultVect<<<numberOfBlocks, threadsPerBlock>>>(X_gpu, Y_gpu, N);
      cudaDeviceSynchronize();

      while(length_reduction > 1)
        {
        sum_reduction_kernel_ext<<<numberOfBlocks, threadsPerBlock>>>(X_gpu, length_reduction);
        cudaDeviceSynchronize();

        length_reduction = ceil(length_reduction/2.);
        }
    
      length_reduction  = N;
      }
    }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

  cout << "   Handmade dot product " << elapsed.count()/(1000.0*A) << " us" << endl;
  
  if(lvlOut)
    {
    cudaMemcpy(dotp_host, X_gpu, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cout << "      Handmade dotp = " << dotp_host[0] << endl;
    }



  /* -----------------------------------------------------
                 cuBLAS - dotProduct
  ----------------------------------------------------- */ 

  initVector<<<numberOfBlocks, threadsPerBlock>>>(X_gpu, Y_gpu, N); // Ré-initialisation de X car il a été modifié par le handmade dotp 
  cudaDeviceSynchronize();

  begin = std::chrono::high_resolution_clock::now(); 

  for(int k=0;k<A;k++)
    {
    for(int i=0; i<M; i++) // Lunching 'M' kernels
      {
      cublasDdot(h_cuBLAS, N, X_gpu, 1, Y_gpu, 1, dotp_gpu);
      cudaDeviceSynchronize();
      }
    }

  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);  

  cout << "   cuBLAS dot product in " << elapsed.count()/(1000.0*A)<< " us" << endl;

  if(lvlOut)
    {
    cudaMemcpy(dotp_host, dotp_gpu, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cout << "      cuBLAS dotp = " << dotp_host[0] << endl;
    }

  cout << "   completed for: " << N << " size, " << M << " repetitions and averaged " << A << " times" << endl;
  
  cudaFree(X_gpu);
  cudaFree(Y_gpu);
  cudaFree(dotp_gpu);
  cudaFree(dotp_host);
  
  return 0 ;
}
