#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "mpi.h"
using namespace std ;

int main(int argc,char**argv)
{
  int rank,size ; 
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  if(rank==0)
  {
    for(int i=0;i<argc;i++)
      cout << argv[i] << " " ;
    cout << endl;
  }

  int lvlOut = 0 ;
  int M = 1 ;
  long N = 10 ;
  int marge = 1 ; 
  double limit = 0.5 ;
  for(int i=0;i<argc;i++)
  {
    if(argv[i]==string("-n"))
      N = stol(argv[i+1]);
    if(argv[i]==string("-m"))
      M = stoi(argv[i+1]);
    if(argv[i]==string("-marge"))
      marge = stoi(argv[i+1]);
    if(argv[i]==string("-limit"))
      limit = stod(argv[i+1]);
    if(argv[i]==string("-out"))
      lvlOut = stoi(argv[i+1]);
  } 

  random_device rd;
  mt19937 mt(rd());
  uniform_real_distribution<double> dist(0.0,1.0); //range is 20 to 22

  long NLoc = N/size ;
  if(rank<N%size)
    NLoc++ ;

  vector<double> X(NLoc) ;
  X.reserve(marge*NLoc);

  for(int i=0;i<NLoc;i++)
    X[i] = dist(mt);

  long LXsize = X.size();
  long GXsize ;
  MPI_Allreduce(&LXsize,&GXsize,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  if(rank==0)
    cout << "X (" << GXsize << ")" << endl;
  
  if(lvlOut>0)
  {
    cout << rank << " X : " ;
    for(const auto & v : X)
      cout << v << " ";
    cout << endl;
  }

  double a,b ;
  a = MPI_Wtime();
  for(int k=0;k<M;k++)
  {
    X.resize(NLoc);  
    for(int i=0;i<X.size();i++)
    {
      if(X[i]>limit)
        X.push_back(dist(mt));
    }
  }
  b = MPI_Wtime();
  LXsize = X.size();
  MPI_Allreduce(&LXsize,&GXsize,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  if(rank==0)
    cout << "X (" << GXsize << ") done in " << b-a << " s"  << endl;
  
  if(lvlOut>0)
  {
    cout << rank << " X : ";
    for(const auto & v : X)
      cout << v << " ";
    cout << endl;
  }

  if(rank==0)
    cout << "completed." << endl;

  MPI_Finalize();
  return 0 ;
}
