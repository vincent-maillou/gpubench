#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <chrono>

#include <Kokkos_Core.hpp>

using namespace std ;

typedef Kokkos::Cuda     DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace>  device_range_policy;

//using view_type = Kokkos::View<double * [3], Kokkos::Device<Kokkos::OpenMP,Kokkos::CudaUVMSpace> >;
//using view_type = Kokkos::View<double * , Kokkos::CudaHostPinnedSpace >;
using view_type = Kokkos::View<double * , Kokkos::CudaUVMSpace >;

struct init_kernel {
  view_type X;
  long N;

  init_kernel(view_type X_, long N_) : X(X_), N(N_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    X(i) = cos(i*2.*M_PI/N);
  }
};

int main(int argc,char**argv)
{
  for(int i=0;i<argc;i++)
    cout << argv[i] << " " ;
  cout << endl;

  bool lvlOut = false;
  int  A = 1;
  int  M = 1;
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
  
  Kokkos::initialize(argc, argv);
  {    
  // Creation des vecteurs
  view_type X("X", N);

  printf("Vector initialization in parallel using kokkos:\n");

  

  /* -----------------------------------------------------
                Initialisation des vecteurs
  ----------------------------------------------------- */


  auto begin = std::chrono::high_resolution_clock::now(); 
  
  for(int k=0;k<A;k++)
    { 
    for(int i=0;i<M;i++)
      {
      Kokkos::parallel_for("X initialization", device_range_policy(0, N), init_kernel(X, N));
      Kokkos::fence();
      }
    }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  
  printf("   Initialisation: %.3f us.\n", elapsed.count()/(1000.0*A) );
  
  
  if(lvlOut)
    {
    cout << "      X: ";
    for(int i=0;i<N;i++)
      {
      cout << X(i) << " ";	      
      }    
    cout << endl;
    }

  cout << "   Completed for: " << N << " size, " << M << " repetitions and averaged " << A  << " times" << endl;
  }
  
  Kokkos::finalize();
  
  return 0 ;
}
