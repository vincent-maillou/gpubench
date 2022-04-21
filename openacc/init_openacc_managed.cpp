#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <chrono>

#include <omp.h>

using namespace std ;

int main(int argc,char**argv)
{
  for(int i=0;i<argc;i++)
    cout << argv[i] << " " ;
  cout << endl;

  bool lvlOut = false;
  int A = 1;
  int M = 1;
  long N = 100000; // Taille des vecteurs
  
  for(int i=0;i<argc;i++) // Parsing des options d'execution
  {
    if(argv[i]==string("-n"))
      N = stol(argv[i+1]);

    if(argv[i]==string("-m"))
      M = stol(argv[i+1]);

    if(argv[i]==string("-a"))
      A = stol(argv[i+1]);

    if(argv[i]==string("-out"))
     lvlOut = true;
  } 
  
    double *X = new double[N];
    
    printf("Vector initialisation in parrallel using openACC managed memory:\n");


    /* -----------------------------------------------------
                  Initialisation des vecteurs
    ----------------------------------------------------- */ 
    auto begin = std::chrono::high_resolution_clock::now(); 

    for(int l=0;l<A;l++)
      {
      for(int k=0;k<M;k++)
        {
        #pragma acc parallel loop    
        for(int i=0;i<N;i++)
          {
          X[i] = cos(i*2.*M_PI/N);
          }   
        }
      }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("   Initialisation: %.3f us.\n", elapsed.count()/(1000.0*A) );

    
    if(lvlOut)
      {
      cout << "   vecteur X: ";
      for(int i = 0; i < N; ++i) 
        {
        cout << X[i] << " ";
        }
      cout << endl;
      }

    delete X;

  
  cout << "   Completed for: " << N << " size, " << M << " repetitions end averaged " << A << " times" << endl;
  return 0 ;
}
