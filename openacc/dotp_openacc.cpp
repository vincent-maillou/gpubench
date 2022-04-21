#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include "mpi.h"

using namespace std ;

int main(int argc,char**argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double start, end, elapsed;

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
  
    long Nloc=N/size;
    if(rank<N%size)
	{
        Nloc++;
        }

    double *X = new double[Nloc];
    double *Y = new double[Nloc];
    double dotp_total = 0.;


    #pragma acc data create(X[0:Nloc], Y[0:Nloc])
    {
        
    if(rank==0)
	{
        printf("Vector manipulation in parrallel using openACC:\n");
        }


    /* -----------------------------------------------------
                  Initialisation des vecteurs
    ----------------------------------------------------- */ 

    long I = 0;
    long offset = rank*Nloc + N%size;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();    

    #pragma acc parallel loop    
    for(int i=0;i<N;i++)
      {
      I = i+rank*Nloc;
      X[i] = cos(I*2.*M_PI/N);
      Y[i] = sin(I*2.*M_PI/N); 
      }   

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    elapsed = end-start;

    if(rank==0)
	{
    	printf("   Initialization: %.3f us.\n", elapsed*1000000.0);
        }
    

    /* -----------------------------------------------------
                       Produit scalaire
    ----------------------------------------------------- */ 
    
    double dotp, dotp_local

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    for(int l=0;l<A;l++)
      {
      for(int k=0;k<M;k++)
        {
        #pragma acc parallel loop reduction(+:dotp_local) 
        for(int i=0;i<Nloc;i++)
          {
          dotp_local += X[i] * Y[i];
          }   
        MPI_Allreduce(&dotp_local, &dotp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        dotp_total += dotp;
        }
      }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    elapsed = end-start;
    
    if(rank==0)
	{
    	printf("   Dotp MPI-openACC: %.3f us.\n", elapsed*1000000.0/A);
  	}
    
    }
    #pragma delete(X) 
    #pragma delete(Y)
    delete X;
    delete Y;
    
    if(lvlOut && rank==0)
      {
      //#pragma acc update self(X[0:N])
      cout << "   dotp: "; cout << dotp_total << " "; cout << endl;
      }
  
  if(rank==0)
	{
  	cout << "   Completed for: " << N << " size, " << M << " repetitions end averaged " << A << " times" << endl;
	}

  MPI_Finalize();
  return 0 ;
}
