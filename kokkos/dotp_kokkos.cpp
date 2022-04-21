#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <chrono>

#include <Kokkos_Core.hpp>

using namespace std ;

typedef Kokkos::Cuda     DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace>  device_range_policy;

using view_type = Kokkos::View<double * , Kokkos::CudaUVMSpace >;


struct init_functor {
  view_type X;
  view_type Y;
  long N;

  init_functor(view_type X_, view_type Y_, long N_) : X(X_), Y(Y_), N(N_) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int i) const {
    X(i) = cos(i*2.*M_PI/N);
    Y(i) = sin(i*2.*M_PI/N);
  }
};


struct dotp_functor {
  view_type X;
  view_type Y;

  dotp_functor(view_type X_, view_type Y_) : X(X_), Y(Y_) {}

  using value_type = double;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, double& dotp) const {
    dotp += X(i) * Y(i);
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
  view_type Y("Y", N);
  double dotp = 0;

  printf("X.Y dot product in parallel using kokkos:\n");

  

  /* -----------------------------------------------------
                Initialisation des vecteurs
  ----------------------------------------------------- */


  Kokkos::parallel_for("X initialization", device_range_policy(0, N), init_functor(X, Y, N));
  Kokkos::fence();
  
 
  /* ----------------------------------------------------
                      X.Y dot product
   ---------------------------------------------------- */

  auto begin = std::chrono::high_resolution_clock::now();

  for(int k=0;k<A;k++)
    {
    for(int i=0;i<M;i++)
      {
      Kokkos::parallel_reduce("X.Y dot product", device_range_policy(0, N), dotp_functor(X, Y), dotp);
      Kokkos::fence();
      }
    }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

  printf("   X.Y dot product: %.3f us.\n", elapsed.count()/(1000.0*A) );
  
  
  if(lvlOut)
    {
    cout << "      dotp =  " << dotp << endl;
    }

  cout << "   Completed for: " << N << " size, " << M << " repetitions and averaged " << A  << " times" << endl;
  }
  
  Kokkos::finalize();
  
  return 0 ;
}
