#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <fstream>
#include "omp.h"

#include <Eigen/Dense>
using namespace Eigen ;

using namespace std ;

//#define EMAT Matrix<double,Dynamic,Dynamic> mat(d,d)
#define EMAT Matrix<double,3,3> mat

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

  int d,D ;
  int nbn,nbe ;
  vector<double> X ;
  vector<int> T ;

  static array<double,6> OneOverFact ;
  //static constexpr array<double,6> OneOverFact = {1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0} ;
};
 
array<double,6> Mesh::OneOverFact = {1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0} ;

Mesh::Mesh(const string & fname)
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


double Mesh::volume()
{
  double vol = 0.0 ; 
  double fac = Mesh::OneOverFact[d] ;
  int d=this->d;
  int D=this->D;
  
  //#pragma omp parallel default(none) shared(X,T,vol) firstprivate(d,D,nbe,fac)
  #pragma omp parallel default(none) shared(vol,nbe) firstprivate(fac,d,D)
  {
    double v = 0.0; 
    //cout << "v= " << v << endl;
    EMAT ;
    //Matrix<double,Dynamic,Dynamic> mat(d,d);
    //vector<double> mat(d*d);
    //vector<double> mat1{{1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0}} ;
    #pragma omp for reduction(+:vol)
    for(int i=0;i<nbe;i++)
    //for(int i=0;i<1680465;i++)
    {
      //cout << omp_get_thread_num() << " " << i << endl;
      int iD = i*D ;
      int n0 = T[iD] ;
      int n0d = n0*d ;
      for(int j=0;j<d;j++)
      {
	int jd = j*d ;
        int n = T[iD+j+1] ;
        int nd = n*d ;
        for(int k=0;k<d;k++)
          mat(j,k) = X[nd+k]-X[n0d+k] ;
	  //mat[jd+k] = X[nd+k]-X[n0d+k] ;
	  //mat[jd+k] = mat1[jd+k] ;
      }
      //v = 1 ; 
      v = fac*mat.determinant();
      vol += v ;
    }
  }
  return vol ;
}

void Mesh::volumesE(vector<double> & volE)
{
  volE.resize(nbe);
  double fac = Mesh::OneOverFact[d] ;
  int d=this->d;
  int D=this->D;
  #pragma omp parallel default(none) shared(volE) firstprivate(d,D,fac)
  {
    double v ;
    EMAT ;
    //Matrix<double,Dynamic,Dynamic> mat(d,d);

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
        for(int k=0;k<d;k++)
          mat(j,k) = X[nd+k]-X[n0d+k] ;
      }
      v = fac*mat.determinant();
      volE[i] = v ;
    }
  }
}

void Mesh::volumesN(vector<double> & volN)
{
  volN.resize(nbn);
  std::fill(volN.begin(),volN.end(),0.0);
  double fac = Mesh::OneOverFact[d] ;
  int d=this->d;
  int D=this->D;

  vector<omp_lock_t> veclocks(nbn);
  #pragma omp parallel for 
  for(int i=0;i<nbn;i++)
    omp_init_lock(&(veclocks[i]));

  #pragma omp parallel default(none) shared(volN,veclocks) firstprivate(d,D,fac)
  {
    double v ;
    EMAT ;
    //Matrix<double,Dynamic,Dynamic> mat(d,d);

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
        for(int k=0;k<d;k++)
          mat(j,k) = X[nd+k]-X[n0d+k] ;
      }
      v = fac*mat.determinant();
      for(int j=0;j<D;j++)
      {
	int n = T[iD+j] ;
	omp_set_lock(&(veclocks[n]));
        volN[n] += v ;
	omp_unset_lock(&(veclocks[n]));
      }
    }
  }
  #pragma omp parallel for 
  for(int i=0;i<nbn;i++)
    omp_destroy_lock(&(veclocks[i]));
}

void Mesh::volumesNavg(vector<double> & volN)
{
  volN.resize(nbn);
  std::fill(volN.begin(),volN.end(),0.0);
  vector<int> nbs(nbn) ;
  double fac = Mesh::OneOverFact[d] ;
  int d=this->d;
  int D=this->D;

  vector<omp_lock_t> veclocks(nbn);
  #pragma omp parallel for 
  for(int i=0;i<nbn;i++)
    omp_init_lock(&(veclocks[i]));

  #pragma omp parallel default(none) shared(volN,nbs,veclocks) firstprivate(d,D,fac)
  {
    double v ;
    EMAT ;
    //Matrix<double,Dynamic,Dynamic> mat(d,d);

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
        for(int k=0;k<d;k++)
          mat(j,k) = X[nd+k]-X[n0d+k] ;
      }
      v = fac*mat.determinant();
      for(int j=0;j<D;j++)
      {
        int n = T[iD+j] ;
        omp_set_lock(&(veclocks[n]));
        volN[n] += v ;
	nbs[n]++ ;
        omp_unset_lock(&(veclocks[n]));
      }
    }
  }
  #pragma omp parallel for 
  for(int i=0;i<nbn;i++)
    omp_destroy_lock(&(veclocks[i]));

  #pragma omp parallel for 
  for(int i=0;i<nbn;i++)
    volN[i] /= nbs[i] ;
}



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

  Eigen::initParallel();
  Mesh m(fname);

  if(lvlOut>1)
  {
    //m.print();
    //m.print(cout,0);
  }
    
  if(Save)
  {
    if(fname[fname.size()-1]=='t')
      m.write(string(fname)+"b");
  }

  double a,b ;

  double vol = 0.0 ;
  a = omp_get_wtime();
  for(int i=0;i<M;i++)
    vol += m.volume() ;
  b = omp_get_wtime();
  cout << "sum vol= " << vol << " done in " << b-a << " s" << endl;

  vector<double> volE(m.nbe) ;
  a = omp_get_wtime();
  for(int i=0;i<M;i++)
    m.volumesE(volE);
  b = omp_get_wtime();
  cout << "volumesE (" << volE.size() << " " << volE[0] << ") done in " << b-a << " s" << endl;
  if(lvlOut>0)
  {
    for(const auto & v : volE)
      cout << v << " ";
    cout << endl;
  }
  vector<double> volN(m.nbn) ;
  a = omp_get_wtime();
  for(int i=0;i<M;i++)
    m.volumesN(volN);
  b = omp_get_wtime();
  cout << "volumesN (" << volN.size() << " " << volN[0] << ") done in " << b-a << " s" << endl;
  if(lvlOut>0)
  {
    for(const auto & v : volN)
      cout << v << " ";
    cout << endl;
  }
  
  a = omp_get_wtime();
  for(int i=0;i<M;i++)
    m.volumesNavg(volN);
  b = omp_get_wtime();
  cout << "volumesNavg (" << volN.size() << " " << volN[0] << ") done in " << b-a << " s" << endl;
  if(lvlOut>0)
  {
    for(const auto & v : volN)
      cout << v << " ";
    cout << endl;
  }
  cout << "completed." << endl;
  //cout << sizeof(omp_lock_t) << endl;
  return 0 ;
}
