
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#define EMAT double mat[3][3] // Cas 3D
// #define EMAT double mat[2][2]

typedef Kokkos::DualView<double*> view_type_double;
typedef Kokkos::DualView<int*> view_type_integer;


template<class ExecutionSpace>
struct Volume {

   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type_double::memory_space, view_type_double::host_mirror_space>::type memory_space;
   /* typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type_integer::memory_space, view_type_integer::host_mirror_space>::type memory_space; */

   Kokkos::View<view_type_double::const_data_type, view_type_double::array_layout, memory_space, Kokkos::MemoryRandomAccess> X;
   Kokkos::View<view_type_integer::const_data_type, view_type_integer::array_layout, memory_space, Kokkos::MemoryRandomAccess> T;

   int d;
   int D;
   double fact;

   Volume (view_type_double X_, view_type_integer T_, int d_, int D_, double fact_)
   {
      d = d_;
      D = D_;
      fact = fact_;

      // Extract the view on the correct Device 
      X = X_.template view<memory_space> ();
      T = T_.template view<memory_space> ();

      // Synchronize the DualView to the correct Device.
      X_.sync<memory_space> ();
      T_.sync<memory_space> ();

      // Mark X and Y as modified.
      X_.modify<memory_space> ();
      T_.modify<memory_space> ();
   }

   KOKKOS_INLINE_FUNCTION
   void operator() (const int i, double & vol) const {
      EMAT;

      int iD = i*D ;
      int n0 = T(iD) ;
      int n0d = n0*d ;

      for(int j(0); j<d; ++j){
         int n = T(iD+j+1) ;
         int nd = n*d ;

         for(int k(0); k<d; ++k){
            mat[j][k] = X(nd+k)-X(n0d+k);
         }
      }

      vol += fact*( mat[0][0]*mat[1][1]*mat[2][2] + mat[0][1]*mat[1][2]*mat[2][0] + mat[0][2]*mat[1][0]*mat[2][1] - mat[0][2]*mat[1][1]*mat[2][0] - mat[0][1]*mat[1][0]*mat[2][2] - mat[0][0]*mat[1][2]*mat[2][1] );
   }
};