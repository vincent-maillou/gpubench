
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>


typedef Kokkos::DualView<double*> view_type;


template<class ExecutionSpace>
struct Add1 {

   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type::memory_space, view_type::host_mirror_space>::type memory_space;

   Kokkos::View<view_type::scalar_array_type, view_type::array_layout, memory_space> X;
   Kokkos::View<view_type::scalar_array_type, view_type::array_layout, memory_space> Y; // , Kokkos::MemoryRandomAccess

   Add1 (view_type X_, view_type Y_)
   {
      // Extract the view on the correct Device 
      X = X_.template view<memory_space> ();
      Y = Y_.template view<memory_space> ();

      // Synchronize the DualView to the correct Device.
      X_.sync<memory_space> ();
      Y_.sync<memory_space> ();

      // Mark X and Y as modified.
      X_.modify<memory_space> ();
      Y_.modify<memory_space> ();
   }

   KOKKOS_INLINE_FUNCTION
   void operator() (const int i) const {
      X(i) += 1;
      Y(i) += 1;
   }
};

template<class ExecutionSpace>
struct Dotp {
   
   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type::memory_space, view_type::host_mirror_space>::type memory_space;

   Kokkos::View<view_type::const_data_type, view_type::array_layout, memory_space, Kokkos::MemoryRandomAccess> X;
   Kokkos::View<view_type::const_data_type, view_type::array_layout, memory_space, Kokkos::MemoryRandomAccess> Y;

   Dotp (view_type X_, view_type Y_)
   {
      // Extract the view on the correct Device 
      X = X_.template view<memory_space> ();
      Y = Y_.template view<memory_space> ();

      // Synchronize the DualView to the correct Device.
      X_.sync<memory_space> ();
      Y_.sync<memory_space> ();

      // Mark X and Y as modified.
      X_.modify<memory_space> ();
      Y_.modify<memory_space> ();
   }

   using value_type = double;

   KOKKOS_INLINE_FUNCTION
   void operator()(const int i, double & sum) const {
      sum += X(i) * Y(i);
   }
};