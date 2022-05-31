#include <iostream>
#include <vector>
#include <cmath>
#include "mpi.h"
using namespace std ;

using entier=long ;

int main(int argc,char **argv)
{
  MPI_Init(&argc,&argv);

  int rank,size ;
  MPI_Comm_rank ( MPI_COMM_WORLD , & rank );
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  entier N = 10 ;
  entier M = 1 ;
  double debut,fin ;
  double start,end ;
  double temps ;

  MPI_Barrier(MPI_COMM_WORLD);
  debut = MPI_Wtime() ;

  for(int i=0;i<argc;++i)
  {
    if(argv[i]==string("-n"))
      N = stol(argv[i+1]);
    if(argv[i]==string("-m"))
      M = stol(argv[i+1]);
  }
  double h=0.5*M_PI/(N-1) ; 
  
  entier Nloc = N / size ; // taille locale des vecteurs 
  
  if(rank==0)
    cout << "sizes : n= " << N << " , nloc= " << Nloc << " , m= " << M << " , h= " << h << endl;

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  vector<double> x(Nloc), y(Nloc) ;

  MPI_Barrier(MPI_COMM_WORLD);
  end= MPI_Wtime();
  temps = end-start ;
  if(rank==0)
     cout << "allocation des vecteurs en "  << temps << " s" << endl;
 
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  entier I ;
  for(entier i=0;i<Nloc;++i)
  {
    I = i + rank*Nloc ;
    x[i]=1.0 ; // pow(cos(I*h),2);
    y[i]=1.0 ; //pow(sin(I*h),2);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();
  temps = end-start ;
  if(rank==0)
    cout << "initialisation des vecteurs en "  << temps << " s" << endl;

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  double ps_tot = 0.0 ;
  for(entier j=0;j<M;++j)
  {
    double ps,ps_loc = 0 ;
    for(entier i=0;i<Nloc;++i)
      ps_loc += x[i]*y[i] ;

    MPI_Allreduce(&ps_loc,&ps,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    ps_tot += ps ;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();
  temps = end-start ;
  if(rank==0)
    cout << "ps_tot= " << ps_tot << " calcule en " << temps << " s"  << endl;

  MPI_Barrier(MPI_COMM_WORLD);
  fin = MPI_Wtime() ;
  temps = fin-debut ;
  if(rank==0)
    cout << "temps total de " << temps << " s"  << endl;

  MPI_Finalize();  
  return 0 ;
}
