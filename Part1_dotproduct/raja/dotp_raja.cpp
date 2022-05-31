#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <chrono>

#include "RAJA/RAJA.hpp"

using namespace std ;

int main(int argc,char**argv)
{
  for(int i=0;i<argc;i++)
    cout << argv[i] << " " ;
  cout << endl;


  bool lvlOut = false;
  int A = 1;
  int M = 1;
  long N = 100000; // Taille des vecteurs
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
  
      
  // Creation des vecteurs
  double *X   = new double[N];
  double *Y   = new double[N];

  double *X_gpu;
  double *Y_gpu;
  cudaMalloc( (void**)&X_gpu, N * sizeof(double) );
  cudaMalloc( (void**)&Y_gpu, N * sizeof(double) );

  RAJA::RangeSegment arange(0, N);

  const int CUDA_BLOCK_SIZE = 256;

  using EXEC_POL3   = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  using REDUCE_POL3 = RAJA::cuda_reduce;

  RAJA::ReduceSum<REDUCE_POL3, double> cuda_sum(0);



  printf("Dot product in parrallel using raja:\n");



  /* -----------------------------------------------------
                Initialisation des vecteurs
  ----------------------------------------------------- */ 
  
  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (int i) {
        X_gpu[i] = cos(i*2.*M_PI/N);
        Y_gpu[i] = sin(i*2.*M_PI/N);
        });
  
  
  if(lvlOut)
  {

  double *X_host;
  double *Y_host;

  cudaMallocHost( (void**)&X_host, N * sizeof(double) );
  cudaMallocHost( (void**)&Y_host, N * sizeof(double) );
  cudaMemcpy( X_host, X_gpu, N * sizeof(double), cudaMemcpyDeviceToHost );
  cudaMemcpy( Y_host, Y_gpu, N * sizeof(double), cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();

  cout << "   Vecteur X: ";
    for(int i = 0; i < N; ++i) 
        {
        cout << X_host[i] << " ";
        }
    cout << endl;
    cout << "   Vecteur Y: ";
    for(int i = 0; i < N; ++i) 
        {
        cout << Y_host[i] << " ";
        }
    cout << endl;
  
  cudaFree(X_host);
  cudaFree(Y_host);
  }



  /*-----------------------------------------------------
                Produit scalaire
  ----------------------------------------------------- */ 

  auto begin = std::chrono::high_resolution_clock::now();  
  
  for(int l=0;l<A;l++)
    {
    for(int k=0;k<M;k++)
      {
        RAJA::forall<EXEC_POL3>(arange, [=] RAJA_DEVICE (int i) {
          cuda_sum += X_gpu[i] * Y_gpu[i];
          });
      }
    }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  
  printf("   Pdt Scalaire: %.3f us.\n", elapsed.count()/(1000.0*A) );
      
  if(lvlOut)
    cout << "   PdtScalaire = " << cuda_sum.get() << endl;

  cudaFree(X_gpu);
  cudaFree(Y_gpu);
  
  cout << "   Completed for: " << N << " size, " << M << " repetitions and averaged " << A << " times" << endl;
  return 0 ;
}
