#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <fstream>
#include <chrono>
#include "omp.h"

using namespace std ;

#define EMAT double mat[3][3] // Cas 3D
// #define EMAT double mat[2][2]

class Mesh
{
public :
  Mesh(const string & fname);

  void print(ostream & os=cout,int decal=1);
  void write(const string & fname);
  void read(const string & fname);

  double volume();
  void volumesE(vector<double> & volE);
  void volumesN(vector<double> & volN);
  void volumesNavg(vector<double> & volN);

  double determinant(double mat) const;
  double determinant(double mat[2][2]) const;
  double determinant(double mat[3][3]) const;

  int d,D ;
  int nbn,nbe ;
  vector<double> X ;
  vector<int> T ;

  const array<double,6> OneOverFact;
};

Mesh::Mesh(const string & fname) : OneOverFact({1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0})
{
  auto ext = fname.substr(fname.find_last_of("."));
  if(ext == ".t")
  {
    ifstream is(fname);
    is >> nbn >> d >> nbe >> D ;
    X.resize(nbn*d);
    for(int i=0;i<X.size();i++)
      is >> X[i] ;
    T.resize(nbe*D);
    for(int i=0;i<T.size();i++)
      is >> T[i] ;

    for(int i=0;i<T.size();i++)
      T[i]--;
    while(T[D*nbe-1]==-1 && nbe>=0)
      nbe-- ;
    T.resize(nbe*D);
  }
  else if(ext==".tb")
   this->read(fname);
}

void Mesh::print(ostream & os,int decal)
{
  os << nbn << " " << d << " " << nbe << " " << D << endl;
  for(int i=0;i<nbn;i++)
  {
    for(int j=0;j<d;j++)
      cout << X[i*d+j] << " ";
    cout << endl;
  }
  for(int i=0;i<nbe;i++)
  {
    for(int j=0;j<D;j++)
      cout << T[i*D+j]+decal << " ";
    cout << endl;
  }
}

void Mesh::write(const string & fname)
{
  ofstream os(fname,ios::binary);
  array<int,4> header ;
  header[0]=d;
  header[1]=D;
  header[2]=nbn;
  header[3]=nbe;
  os.write(reinterpret_cast<char*>(&(header[0])),sizeof(header));
  os.write(reinterpret_cast<char*>(&(X[0])),X.size()*sizeof(double));
  os.write(reinterpret_cast<char*>(&(T[0])),T.size()*sizeof(int));
}

void Mesh::read(const string & fname)
{
  ifstream is(fname,ios::binary);
  array<int,4> header ;
  is.read(reinterpret_cast<char*>(&(header[0])),sizeof(header));
  d = header[0] ;
  D = header[1] ;
  nbn= header[2] ;
  nbe = header[3] ;
  X.resize(nbn*d);
  is.read(reinterpret_cast<char*>(&(X[0])),X.size()*sizeof(double));
  T.resize(nbe*D);
  is.read(reinterpret_cast<char*>(&(T[0])),T.size()*sizeof(int));
}



/* -------------------------------------------
                  MESH::volume()
   ------------------------------------------- */

double Mesh::volume()
{
  double vol = 0.0 ; 
  
  int d=this->d;
  int D=this->D;
  
  #pragma omp parallel default(none) shared(vol,nbe) firstprivate(d,D)
  {
    double v = 0.0; 
    EMAT ;

    #pragma omp for reduction(+:vol)
    for(int i=0;i<nbe;i++){
      int iD = i*D ;
      int n0 = T[iD] ;
      int n0d = n0*d ;

      for(int j=0;j<d;j++){
	      int jd = j*d ;
        int n = T[iD+j+1] ;
        int nd = n*d ;

        for(int k=0;k<d;k++){
          mat[j][k] = X[nd+k]-X[n0d+k] ;
        }
      }
      
      vol += determinant(mat);
    }
  }

  return vol ;
}



/* -------------------------------------------
                  MESH::volumesE
   ------------------------------------------- */

void Mesh::volumesE(vector<double> & volE)
{
  volE.resize(nbe);
  
  int d=this->d;
  int D=this->D;

  #pragma omp parallel default(none) shared(volE) firstprivate(d,D)
{
  double v ;
  EMAT ;

  #pragma omp for 
  for(int i=0;i<nbe;i++)
  {
    int iD = i*D ;
    int n0 = T[iD] ;
    int n0d = n0*d ;

    for(int j=0;j<d;j++)
    {
      int n = T[iD+j+1] ;
      int nd = n*d ;

      for(int k=0;k<d;k++){
        mat[j][k] = X[nd+k]-X[n0d+k] ;
      }
    }
    
    v = determinant(mat);
    volE[i] = v ;
    }
  }
}



/* -------------------------------------------
                  MESH::volumesN()
   ------------------------------------------- */

void Mesh::volumesN(vector<double> & volN)
{
  volN.resize(nbn);
  std::fill(volN.begin(),volN.end(),0.0);

  int d=this->d;
  int D=this->D;
  
 #pragma omp parallel default(none) shared(volN) firstprivate(d,D)
  {
    double v ;
    EMAT ;

    #pragma omp for 
    for(int i=0;i<nbe;i++)
    {
      int iD = i*D ;
      int n0 = T[iD] ;
      int n0d = n0*d ;

      for(int j=0;j<d;j++){
        int n = T[iD+j+1] ;
        int nd = n*d ;

        for(int k=0;k<d;k++){
          mat[j][k] = X[nd+k]-X[n0d+k] ;
        }
      }
      
      v = determinant(mat);

      for(int j=0;j<D;j++){
        #pragma omp atomic 
        volN[T[iD+j]] += v ;
      }
    }
  }

}



/* -------------------------------------------
              MESH::volumesNavg()
   ------------------------------------------- */

void Mesh::volumesNavg(vector<double> & volN)
{
  volN.resize(nbn);
  std::fill(volN.begin(),volN.end(),0.0);
  vector<int> nbs(nbn);
  
  int d=this->d;
  int D=this->D;

#pragma omp parallel default(none) shared(volN,nbs) firstprivate(d,D)
  {
    double v ;
    EMAT ;

    #pragma omp for 
    for(int i=0;i<nbe;i++)
    {
      int iD = i*D ;
      int n0 = T[iD] ;
      int n0d = n0*d ;

      for(int j=0;j<d;j++){
        int n = T[iD+j+1] ;
        int nd = n*d ;

        for(int k=0;k<d;k++){
          mat[j][k] = X[nd+k]-X[n0d+k] ;
        }
      }

      v = determinant(mat);

      for(int j=0;j<D;j++){
        #pragma omp atomic 
        volN[T[iD+j]] += v ;
      }
        
      for(int j=0;j<D;j++){
        #pragma omp atomic 
        nbs[T[iD+j]]++ ;
      }
        
    }

    #pragma omp for 
    for(int i=0;i<nbn;i++){
      volN[i] /= nbs[i] ;
    }
  }
}



/* -------------------------------------------
                  MESH::utilities
   ------------------------------------------- */

double Mesh::determinant(double mat) const{
  return OneOverFact[1]*mat;
}

double Mesh::determinant(double mat[2][2]) const{
  return OneOverFact[2]*(mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]);
}

double Mesh::determinant(double mat[3][3]) const{
  return OneOverFact[3]*(mat[0][0]*mat[1][1]*mat[2][2] + mat[0][1]*mat[1][2]*mat[2][0] + mat[0][2]*mat[1][0]*mat[2][1] - mat[0][2]*mat[1][1]*mat[2][0] - mat[0][1]*mat[1][0]*mat[2][2] - mat[0][0]*mat[1][2]*mat[2][1]);
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
  int Save = 0 ;
  string fname ;
  for(int i=0;i<argc;i++)
  {
    if(argv[i]==string("-mesh"))
      fname = argv[i+1];
    if(argv[i]==string("-m"))
      M = stoi(argv[i+1]);
    if(argv[i]==string("-save"))
      Save = stoi(argv[i+1]);
    if(argv[i]==string("-out"))
      lvlOut = stoi(argv[i+1]);
  } 
  
  Mesh m(fname);

  /* if(lvlOut>1)
  {
    m.print();
    //m.print(cout,0);
    if(fname[fname.size()-1]=='t')
      m.write(string(fname)+"b");
  } */



  /* ---------computation of volume()--------- */

  auto begin = std::chrono::high_resolution_clock::now();  

    double vol = 0.0 ;
    for(int i=0;i<M;i++)
      vol += m.volume() ;

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);


  cout << endl << "Mesh volume() computation:" << endl;
  cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;
  cout << "   volume = " << vol << endl;



  /* ---------computation of volumesE()--------- */

  vector<double> volE(m.nbe) ;

  begin = std::chrono::high_resolution_clock::now();  

    for(int i=0;i<M;i++)
      m.volumesE(volE);

  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

  cout << endl << "Mesh volumesE() computation:" << endl;
  cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

  if(lvlOut >= 1){
    switch(lvlOut){
      case 1:
        cout << "   data: ";
        for(auto elem : volE){
          cout << elem << " ";
        }
        cout << endl;
        break;
      case 2:
        ofstream ofs("volE_serial.txt");
        for(auto elem : volE){
          ofs << elem << endl;
        }
        break;
    } 
  }



  /* ---------computation of volumesN()--------- */

  vector<double> volN(m.nbn) ;

  begin = std::chrono::high_resolution_clock::now();  

    for(int i=0;i<M;i++)
      m.volumesN(volN); 

  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

  cout << endl << "Mesh volumesN() computation:" << endl;
  cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

  if(lvlOut >= 1){
    switch(lvlOut){
      case 1:
        cout << "   data: ";
        for(auto elem : volN){
          cout << elem << " ";
        }
        cout << endl;
        break;
      case 2:
        ofstream ofs("volN_serial.txt");
        for(auto elem : volN){
          ofs << elem << endl;
        } 
      break;
    }  
  }
  


  /* ---------computation of volumesNavg()--------- */

  vector<double> volNavg(m.nbn);

  begin = std::chrono::high_resolution_clock::now();  

    for(int i=0;i<M;i++)
      m.volumesNavg(volNavg);
  
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

  cout << endl << "Mesh volumesNavg() computation: " << endl;
  cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

  if(lvlOut >= 1){
    switch(lvlOut){
      case 1:
        cout << "   data: ";
        for(auto elem : volNavg){
          cout << elem << " ";
        }
        cout << endl;
        break;
      case 2:
        ofstream ofs("volNavg_serial.txt");
        for(auto elem : volNavg){
          ofs << elem << endl;
        } 
      break;
    }  
  }



  cout << endl << "completed." << endl;
  return 0 ;
}
