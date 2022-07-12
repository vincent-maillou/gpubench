
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#define EMAT double mat[3][3] // Cas 3D
//#define EMAT double mat[2][2]

typedef Kokkos::DualView<double*> view_type_double;
typedef Kokkos::DualView<int*> view_type_integer;



/* -------------------------------------------
               class Utils
   ------------------------------------------- */

class Utils {
   public:
      KOKKOS_INLINE_FUNCTION
      double determinant(double mat) const{ return mat; }
      KOKKOS_INLINE_FUNCTION
      double determinant(double mat[2][2]) const{ return (mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]); }
      KOKKOS_INLINE_FUNCTION
      double determinant(double mat[3][3]) const{ return (mat[0][0]*mat[1][1]*mat[2][2] + mat[0][1]*mat[1][2]*mat[2][0] + mat[0][2]*mat[1][0]*mat[2][1] - mat[0][2]*mat[1][1]*mat[2][0] - mat[0][1]*mat[1][0]*mat[2][2] - mat[0][0]*mat[1][2]*mat[2][1]); }
};



/* -------------------------------------------
               struct Volume
   ------------------------------------------- */

template<class ExecutionSpace>
struct Volume : public Utils{

   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type_double::memory_space, view_type_double::host_mirror_space>::type memory_space;

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

      vol += fact*Utils::determinant(mat);
   }
};



/* -------------------------------------------
               struct VolumesE
   ------------------------------------------- */

template<class ExecutionSpace>
struct VolumesE : public Utils{

   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type_double::memory_space, view_type_double::host_mirror_space>::type memory_space;

   Kokkos::View<view_type_double::const_data_type, view_type_double::array_layout, memory_space, Kokkos::MemoryRandomAccess> X;
   Kokkos::View<view_type_integer::const_data_type, view_type_integer::array_layout, memory_space, Kokkos::MemoryRandomAccess> T;

   Kokkos::View<view_type_double::scalar_array_type, view_type_double::array_layout, memory_space> volE;

   int d;
   int D;
   double fact;

   VolumesE (view_type_double X_, view_type_integer T_, view_type_double volE_, int d_, int D_, double fact_)
   {
      d = d_;
      D = D_;
      fact = fact_;

      // Extract the view on the correct Device 
      X = X_.template view<memory_space> ();
      T = T_.template view<memory_space> ();
      volE = volE_.template view<memory_space> ();

      // Synchronize the DualView to the correct Device.
      X_.sync<memory_space> ();
      T_.sync<memory_space> ();
      volE_.sync<memory_space> ();

      // Mark X and Y as modified.
      X_.modify<memory_space> ();
      T_.modify<memory_space> ();
      volE_.modify<memory_space> ();
   }

   KOKKOS_INLINE_FUNCTION
   void operator() (const int i) const {
      double v(0.);
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

      v = fact*Utils::determinant(mat);
      volE(i) = v;
   }
};



/* -------------------------------------------
               struct VolumesN
   ------------------------------------------- */

template<class ExecutionSpace>
struct VolumesN : public Utils{

   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type_double::memory_space, view_type_double::host_mirror_space>::type memory_space;

   Kokkos::View<view_type_double::const_data_type, view_type_double::array_layout, memory_space, Kokkos::MemoryRandomAccess> X;
   Kokkos::View<view_type_integer::const_data_type, view_type_integer::array_layout, memory_space, Kokkos::MemoryRandomAccess> T;

   Kokkos::View<view_type_double::scalar_array_type, view_type_double::array_layout, memory_space> volN;

   int d;
   int D;
   double fact;

   VolumesN (view_type_double X_, view_type_integer T_, view_type_double volN_, int d_, int D_, double fact_)
   {
      d = d_;
      D = D_;
      fact = fact_;

      // Extract the view on the correct Device 
      X = X_.template view<memory_space> ();
      T = T_.template view<memory_space> ();
      volN = volN_.template view<memory_space> ();

      // Synchronize the DualView to the correct Device.
      X_.sync<memory_space> ();
      T_.sync<memory_space> ();
      volN_.sync<memory_space> ();

      // Mark X and Y as modified.
      X_.modify<memory_space> ();
      T_.modify<memory_space> ();
      volN_.modify<memory_space> ();
   }

   KOKKOS_INLINE_FUNCTION
   void operator() (const int i) const {
      double v(0.);
      EMAT ;

      int iD = i*D;
      int n0 = T(iD);
      int n0d = n0*d;

      for(int j=0;j<d;j++)
      {
         int n = T(iD+j+1) ;
         int nd = n*d ;

         for(int k=0;k<d;k++){
            mat[j][k] = X(nd+k)-X(n0d+k);
         }
      }

      v = fact*Utils::determinant(mat);
      
      for(int j=0;j<D;j++){
         Kokkos::atomic_add(&volN(T(iD+j)), v);
      }
   }
};



/* -------------------------------------------
               struct VolumesNavg
   ------------------------------------------- */

template<class ExecutionSpace>
struct VolumesNavg : public Utils{

   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type_double::memory_space, view_type_double::host_mirror_space>::type memory_space;

   Kokkos::View<view_type_double::const_data_type, view_type_double::array_layout, memory_space, Kokkos::MemoryRandomAccess> X;
   Kokkos::View<view_type_integer::const_data_type, view_type_integer::array_layout, memory_space, Kokkos::MemoryRandomAccess> T;

   Kokkos::View<view_type_double::scalar_array_type, view_type_double::array_layout, memory_space> volNavg;
   Kokkos::View<view_type_integer::scalar_array_type, view_type_integer::array_layout, memory_space> pnbs;

   int d;
   int D;
   double fact;

   VolumesNavg (view_type_double X_, view_type_integer T_, view_type_double volNavg_, view_type_integer pnbs_, int d_, int D_, double fact_)
   {
      d = d_;
      D = D_;
      fact = fact_;

      // Extract the view on the correct Device 
      X = X_.template view<memory_space> ();
      T = T_.template view<memory_space> ();
      volNavg = volNavg_.template view<memory_space> ();
      pnbs = pnbs_.template view<memory_space> ();

      // Synchronize the DualView to the correct Device.
      X_.sync<memory_space> ();
      T_.sync<memory_space> ();
      volNavg_.sync<memory_space> ();
      pnbs_.sync<memory_space> ();

      // Mark X and Y as modified.
      X_.modify<memory_space> ();
      T_.modify<memory_space> ();
      volNavg_.modify<memory_space> ();
      pnbs_.modify<memory_space> ();
   }

   KOKKOS_INLINE_FUNCTION
   void operator() (const int i) const {
      double v(0.);
      EMAT ;

      int iD = i*D;
      int n0 = T(iD);
      int n0d = n0*d;

      for(int j=0;j<d;j++)
      {
         int n = T(iD+j+1) ;
         int nd = n*d ;

         for(int k=0;k<d;k++){
            mat[j][k] = X(nd+k)-X(n0d+k);
         }
      }

      v = fact*Utils::determinant(mat);
      
      for(int j=0;j<D;j++){
         Kokkos::atomic_add(&volNavg(T(iD+j)), v);
         Kokkos::atomic_increment(&pnbs(T(iD+j)));
      }
   }
};



/* -------------------------------------------
               struct Avg
   ------------------------------------------- */

template<class ExecutionSpace>
struct Avg : public Utils{

   typedef ExecutionSpace execution_space;

   typedef typename std::conditional<std::is_same<ExecutionSpace,Kokkos::DefaultExecutionSpace>::value ,
         view_type_double::memory_space, view_type_double::host_mirror_space>::type memory_space;

   Kokkos::View<view_type_double::scalar_array_type, view_type_double::array_layout, memory_space> volNavg;
   Kokkos::View<view_type_integer::scalar_array_type, view_type_integer::array_layout, memory_space> pnbs;

   Avg (view_type_double volNavg_, view_type_integer pnbs_)
   {
      // Extract the view on the correct Device 
      volNavg = volNavg_.template view<memory_space> ();
      pnbs = pnbs_.template view<memory_space> ();

      // Synchronize the DualView to the correct Device.
      volNavg_.sync<memory_space> ();
      pnbs_.sync<memory_space> ();

      // Mark X and Y as modified.
      volNavg_.modify<memory_space> ();
      pnbs_.modify<memory_space> ();
   }

   KOKKOS_INLINE_FUNCTION
   void operator() (const int i) const {
      volNavg(i) = volNavg(i)/pnbs(i);
   }
};