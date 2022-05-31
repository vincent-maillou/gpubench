#include <iostream>
#include <vector>
#include <cmath>
using namespace std ;

int main(int argc,char**argv)
{
  for(int i=0;i<argc;i++)
    cout << argv[i] << " " ;
  cout << endl;

  int lvlOut = 0 ;
  int M = 1 ;
  long N = 10 ;
  for(int i=0;i<argc;i++)
  {
    if(argv[i]==string("-n"))
      N = stol(argv[i+1]);
    if(argv[i]==string("-m"))
      M = stol(argv[i+1]);
    if(argv[i]==string("-out"))
      lvlOut = stoi(argv[i+1]);
  } 

  vector<double> X(N), Y(N) ;

  for(int i=0;i<N;i++)
    X[i] = 1.0/(i+1) ;
  for(int i=0;i<N;i++)
    Y[i] = cos(i/(2*M_PI)) ;

  if(lvlOut>1)
  {
    for(const auto & v : X)
      cout << v << " ";
    cout << endl;
    for(const auto & v : Y)
      cout << v << " ";
    cout << endl;
  }
  
  double ps = 0.0 ;
  for(int k=0;k<M;k++)
  {
    for(int i=0;i<N;i++)
      ps += X[i]*Y[i] ;
  }
  if(lvlOut>0)
    cout << "ps= " << ps << endl;

  double a = 3.14 ;
  for(int k=0;k<M;k++)
  {
    for(int i=0;i<N;i++)
      Y[i] += a*X[i] ;
  }
  if(lvlOut>1)
  {
    for(const auto & v : Y)
      cout << v << " ";
    cout << endl;
  }

  cout << "completed." << endl;
  return 0 ;
}
