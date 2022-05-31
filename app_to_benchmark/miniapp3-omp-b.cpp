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
    X.resize(marge*N); // max size.

#pragma omp parallel firstprivate(limit,marge)
    {
      int Nloc = 0 ;
      int nbt2 = omp_get_num_threads() ;
      int idt = omp_get_thread_num() ;
      int NI = N + idt ;  
      random_device rd2;
      mt19937 mt2(rd2());
      uniform_real_distribution<double> dist2(0.0,1.0); //range is 20 to 22
#pragma omp for nowait 
      for(int i=0;i<N;i++)
      {
        if(X[i]>limit)
	{
          X[NI+Nloc*nbt2]=dist2(mt2);
	  Nloc++;
	}
      }
      for(int i=0;i<Nloc;i++)
      {
        if(X[NI+i*nbt2]>limit)
	{
          X[NI+Nloc*nbt2]=dist2(mt2);
	  Nloc++;
	}
      }
      ExtraNbs[idt] = Nloc ;
#pragma omp barrier
#pragma omp single
      {
        ExtraPos[0] = 0 ;
	for(int i=0;i<nbt2;i++)
	  ExtraPos[i+1]=ExtraPos[i]+ExtraNbs[i] ;
      }
#pragma omp barrier
    }

    cout << "ExtraPos:" << endl;
    for(int i=0;i<=nbt;i++)
      cout << ExtraPos[i] << " ";
    cout << endl;
    
    int NN=N+ExtraPos[nbt];
    vector<int> FreePos ; 
    int Navg = ExtraPos[nbt]/nbt ;
    if(ExtraPos[nbt]%nbt)
      Navg++;

    cout << "Navg= " << Navg << endl;
    for(int i=0;i<nbt;i++)
    {
      for(int j=(ExtraPos[i+1]-ExtraPos[i]);j<Navg;j++)
        FreePos.push_back(N+i+j*nbt);
    }
    /*
    for(int i=0;i<X.size();i++)
      cout << i<<","<< X[i]<< " ";
    cout << endl;
    */
    cout << "FreePos: " << FreePos.size()  << endl;
    /*
    for(const auto & v : FreePos)
      cout << v << " " ;
    cout << endl;
    */
    int pos=0;
    for(int i=0;i<nbt;i++)
    {
      for(int j=Navg-1;j<(ExtraPos[i+1]-ExtraPos[i]);j++)
      {
	int p = N+i+j*nbt ;
        if(p>=NN)
	{
	  X[FreePos[pos]]=X[p];
	  pos++;
	}
      }
    }
    X.resize(NN);
    /*
    for(int i=0;i<X.size();i++)
      cout << i<<","<< X[i]<< " ";
    cout << endl;
    */
  }
  if(lvlOut>1)
  {
    cout << "ExtraPos : " ; 
    for(const auto & v : ExtraPos)
      cout << v << " ";
    cout << endl;
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
