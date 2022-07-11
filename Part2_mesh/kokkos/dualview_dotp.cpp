
#include <ostream>
#include "functors_dotp.hpp"


void init(view_type X, view_type Y);
void add1(view_type X, view_type Y);
void dotp(view_type X, view_type Y, double & sum);

int main (int narg, char* arg[]) {
  
  std::cout << "initializing kokkos....." <<std::endl;

   Kokkos::initialize (narg, arg);
   
   std::cout << "......done." << std::endl;
   {
      // Create DualViews. This will allocate on both the device and its
      // host_mirror_device.

      const int size = 10000;

      view_type X ("X vector", size);
      view_type Y ("Y vector", size);

      init(X, Y);
      std::cout << "Init: X[0]=" << X.h_view(0) << std::endl;
      add1(X, Y);
      std::cout << "Add: X[0]=" << X.h_view(0) << std::endl;

      double sum(0.);
      dotp(X, Y, sum);
      std::cout << "Dotp: dotp=" << sum << std::endl;

   }

   Kokkos::finalize();
}

void init(view_type X, view_type Y){
   // Init on CPU 
   view_type::t_host h_X = X.h_view;
   view_type::t_host h_Y = Y.h_view;

   for (view_type::size_type i(0); i < h_X.extent(0); ++i) {
      h_X(i) = 0;
      h_Y(i) = 0;
   }

   // Mark as modified and mirror space to copy data on GPU
   X.modify<view_type::host_mirror_space> ();
   Y.modify<view_type::host_mirror_space> ();
}

void add1(view_type X, view_type Y){
   // Run on the device.  This will cause data movement to the device,
   // since it was marked as modified on the host.

   const int size = X.extent(0);

   Kokkos::parallel_for(size, Add1<view_type::execution_space>(X, Y));
   Kokkos::fence();

   // Sync Device/Host (Transfer back data from device to host for verification)
   X.sync<view_type::host_mirror_space> ();
   Y.sync<view_type::host_mirror_space> ();
}

void dotp(view_type X, view_type Y, double & sum){

   const int size = X.extent(0);

   Kokkos::parallel_reduce(size, Dotp<view_type::execution_space>(X, Y), sum);
   Kokkos::fence();
}