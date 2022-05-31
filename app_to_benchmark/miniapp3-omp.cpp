#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "omp.h"
using namespace std ;

int main(int argc,char**argv)
{
  for(int i=0;i<argc;i++)
    cout << argv[i] << " " ;
  cout << endl;

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

  vector<double> X(N) ;
  X.reserve(marge*N);

  for(int i=0;i<N;i++)
    X[i] = dist(mt);

  cout << "X (" << X.size() << ")" << endl;
  if(lvlOut>0)
  {
    for(const auto & v : X)
      cout << v << " ";
    cout << endl;
  }

  int nbt ;
  #pragma omp parallel 
  {
    #pragma omp single
      nbt = omp_get_num_threads() ;
  }
  cout << "Nb Threads= " << nbt << endl;
  vector<int> ExtraNbs(nbt);
  vector<int> ExtraPos(nbt+1);

  double a,b ;
  a = omp_get_wtime();
  for(int k=0;k<M;k++)
  {
    X.resize(N);  

#pragma omp parallel firstprivate(limit,marge)
    {
      int nbt2 = omp_get_num_threads() ;
      int idt = omp_get_thread_num() ;
      random_device rd2;
      mt19937 mt2(rd2());
      uniform_real_distribution<double> dist2(0.0,1.0); //range is 20 to 22
      vector<double> XT ;
      XT.reserve((marge-1)*N);
#pragma omp for nowait 
      for(int i=0;i<N;i++)
      {
        if(X[i]>limit)
          XT.push_back(dist2(mt2));
      }
      for(int i=0;i<XT.size();i++)
      {
        if(XT[i]>limit)
          XT.push_back(dist2(mt2));
      }
      //cout << "XT (" << XT.size() << ")" << endl ;
      ExtraNbs[idt] = XT.size() ;
#pragma omp barrier
#pragma omp single
      {
        ExtraPos[0] = 0 ;
	for(int i=0;i<nbt2;i++)
	  ExtraPos[i+1]=ExtraPos[i]+ExtraNbs[i] ;
        X.resize(N+ExtraPos[nbt2]);	
      }
#pragma omp barrier
      for(int i=0;i<XT.size();i++)
        X[N+ExtraPos[idt]+i]=XT[i] ;
    }
  }
  b = omp_get_wtime();
  cout << "X (" << X.size() << ") done in " << b-a << " s"  << endl;
  if(lvlOut>0)
  {
    for(const auto & v : X)
      cout << v << " ";
    cout << endl;
  }

  cout << "completed." << endl;
  return 0 ;
}
