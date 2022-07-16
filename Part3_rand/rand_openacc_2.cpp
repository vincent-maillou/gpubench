#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#include "curand.h"

using namespace std ;


/*  WORK IN PROGRESS..

  1. How to resize a vector on the GPU
    - #pragma acc shape[( shape-name-string )] shape-clause-list


  2. How to generate random number in parallel on the GPU
    To Compile: nvc++ -fast -acc=gpu -cudalib=curand -o trand6.exe trand6.cpp

    cuRAND: https://docs.nvidia.com/cuda/curand/introduction.html#introduction
    CPU-side generation: /include/curand.h
    GPU-side generation: /include/curand_kernel.h

    /opt/nvidia/hpc_sdk/Linux_x86_64/22.2/math_libs/include/curand_kernel.h
    For exemples: /opt/nvidia/hpc_sdk/Linux_x86_64/22.2/examples/CUDA-Libraries/cuRAND

    Examples intéréssants:
      make cpp_test6       : OpenACC C++ with host calls
      make cpp_test7       : OpenACC C++ calls within data regions
      make cpp_test8       : OpenACC C++ calls in compute regions
*/


/* struct rollingContainer{
  vector<float> data;
  float *restrict pData;

  rollingContainer(size_t n, int marge){
    data.reserve(marge*n);
    pData = &data[0];
    tData = data.size();
  }

  ~rollingContainer(){
    pData = nullptr;
  }

  size_t getSize() const{ return tData; }
  size_t getReserve() const{ return reservedSpace; }

  void resize(size_t n){
    data.resize(n);
    tData = n;
    reservedSpace = data.capacity();
  }

  void reserve(size_t n, int marge){
    data.reserve(n*marge);
    reservedSpace = n*marge;
  }

  void shrink_to_fit(){
    data.shrink_to_fit();
    reservedSpace = data.size();
  }

  private:
    size_t tData;
    size_t reservedSpace;
}; */

void resizeOnGPU(float * pCont, size_t tCont, size_t & capCont, int marge){
  float * pTemp = new float[tCont*marge];
  size_t capTemp = tCont*marge;

  // Copy data from the old container to the new one
  #pragma acc enter data create(pTemp[:capTemp]) 
  
  #pragma acc data present(pCont[capCont], pTemp[:capTemp])
  #pragma acc parallel loop
  for(size_t i(0); i<tCont; ++i){
    pTemp[i] = pCont[i];
  }

  #pragma acc exit data delete(pCont[:capCont]) // Delete old container on GPU
  delete pCont; // Delete old container on CPU
  pCont = pTemp; // Change pointer to new container
  pTemp = nullptr;
  capCont = capTemp; // Update container reserved capacity
}



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

  curandGenerator_t cuGen;

  // Create the random generator on the host and set the seed
  curandCreateGeneratorHost(&cuGen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(cuGen, time(NULL));

  vector<float> X(N);
  float *restrict pX(&X[0]);
  size_t tX(X.size());

  X.reserve(marge*N);
  size_t capX(X.capacity());

  // Generate the uniform distribution and fill X vector
  curandGenerateUniform(cuGen, &X[0], X.size());

  cout << "X (" << X.size() << ")" << endl;
  if(lvlOut == 1){
    for(const auto & v : X)
      cout << v << " ";
    cout << endl;
  }



  /* ---------modification of X vector--------- */
  
  auto begin = std::chrono::high_resolution_clock::now();  


  #pragma acc enter data copyin(pX[:capX])

  #pragma acc data present(pX[:capX])
  #pragma acc loop seq 
  unsigned pushedBack(0);
  for(size_t i(0); i<tX; ++i){

    if(tX + pushedBack == capX){ // Max capacity filled, need a resize
      resizeOnGPU(pX, tX, capX, marge);
    }

    if(pX[i] > limit){
      pX[tX + i] = i;
    }
  }

  #pragma acc update self(pX[:capX])
  #pragma acc exit data delete(pX[:capX])


/*     for(int k=0;k<M;k++){

      X.resize(N);  
      X.shrink_to_fit();
      X.reserve(marge*N);

      for(int i=0;i<X.size();i++){
        if(X[i]>limit){
          float temp(0.);
          curandGenerateUniform(cuGen, &temp, 1);
          X.push_back(temp);
        }
      }
    } */

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  
  cout << "X (" << X.size() << ")" << endl;
  cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

  if(lvlOut == 1){
    for(const auto & v : X)
      cout << v << " ";
    cout << endl;
  }
  else if(lvlOut == 2){
    double means(0.);
    for(const auto & v : X){
      means += v;
    }
    means /= X.size();
    cout << "Uniforme distribution around: " << means << endl;
  }

  // Destroy the random generator
  curandDestroyGenerator(cuGen);



  cout << "completed." << endl;
  return 0 ;
}
