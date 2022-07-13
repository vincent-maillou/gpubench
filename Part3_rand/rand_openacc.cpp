#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

using namespace std ;


/*
  1. How to resize a vector on the GPU
    - #pragma acc shape[( shape-name-string )] shape-clause-list


  2. How to generate random number in parallel on the GPU
    - Using random number generator require to compile with: "nollvm"
    - -acc -ta=tesla:nollvm -Mcudalib=curand
*/


/* -------------------------------------------
                  MAIN
   ------------------------------------------- */

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

  vector<double> X(N);
  X.reserve(marge*N);

  for(int i=0;i<N;i++)
    X[i] = dist(mt);

  cout << "X (" << X.size() << ")" << endl;
  if(lvlOut>0){
    for(const auto & v : X)
      cout << v << " ";
    cout << endl;
  }



  /* ---------modification of X vector--------- */
  
  auto begin = std::chrono::high_resolution_clock::now();  

    for(int k=0;k<M;k++){

      X.resize(N);  
      X.shrink_to_fit();
      X.reserve(marge*N);

      for(int i=0;i<X.size();i++){
        if(X[i]>limit)
          X.push_back(dist(mt));
      }
    }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  
  cout << "X (" << X.size() << ")" << endl;
  cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

  if(lvlOut>0){
    for(const auto & v : X)
      cout << v << " ";
    cout << endl;
  }

  cout << "completed." << endl;
  return 0 ;
}
