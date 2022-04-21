#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <chrono>


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
    double *Y = new double[N];
    double dotp = 0.;
    #pragma acc data create(X[0:N], Y[0:N])
    {
        
    printf("Vector manipulation in parrallel using openACC:\n");


    /* -----------------------------------------------------
                  Initialisation des vecteurs
    ----------------------------------------------------- */ 

    #pragma acc parallel loop    
    for(int i=0;i<N;i++)
      {
      X[i] = cos(i*2.*M_PI/N);
      Y[i] = sin(i*2.*M_PI/N); 
      }   

    /* -----------------------------------------------------
                  Initialisation des vecteurs
    ----------------------------------------------------- */ 
    
    auto begin = std::chrono::high_resolution_clock::now();     

    for(int l=0;l<A;l++)
      {
      for(int k=0;k<M;k++)
        {
        #pragma acc parallel loop reduction(+:dotp) 
        for(int i=0;i<N;i++)
          {
          dotp += X[i] * Y[i];
          }   
        }
      }


    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("   openAcc dot product: %.3f us.\n", elapsed.count()/(1000.0*A) );
    
    }
    #pragma delete(X) 
    #pragma delete(Y)
    delete X;
    delete Y;
    
    if(lvlOut)
      {
      //#pragma acc update self(X[0:N])
      cout << "   dotp: "; cout << dotp << " "; cout << endl;
      }
  
  cout << "   Completed for: " << N << " size, " << M << " repetitions end averaged " << A << " times" << endl;
  return 0 ;
}
