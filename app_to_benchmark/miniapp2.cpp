#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <fstream>
#include <ctime>

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
  double v ; 
  EMAT ;
  //Matrix<double,Dynamic,Dynamic> mat(d,d);
  //Matrix<double,Dynamic,Dynamic,RowMajor> mat(d,d);
  
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
    v = Mesh::OneOverFact[d]*mat.determinant();
    vol += v ;
  }
  return vol ;
}

void Mesh::volumesE(vector<double> & volE)
{
  double v ;
  EMAT ;
  //Matrix<double,Dynamic,Dynamic> mat(d,d);
  volE.resize(nbe);

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
    v = Mesh::OneOverFact[d]*mat.determinant();
    volE[i] = v ;
  }
}

void Mesh::volumesN(vector<double> & volN)
{
  double v ;
  EMAT ;
  //Matrix<double,Dynamic,Dynamic> mat(d,d);
  volN.resize(nbn);
  std::fill(volN.begin(),volN.end(),0.0);

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
    v = Mesh::OneOverFact[d]*mat.determinant();
    for(int j=0;j<D;j++)
      volN[T[iD+j]] += v ;
  }
}

void Mesh::volumesNavg(vector<double> & volN)
{
  double v ;
  EMAT ;
  //Matrix<double,Dynamic,Dynamic> mat(d,d);
  volN.resize(nbn);
  std::fill(volN.begin(),volN.end(),0.0);
  vector<int> nbs(nbn);

  for(int i=0;i<nbe;i++)
  {
    int iD = i*D ;
    int n0 = T[iD] ;
    int n0d = n0*d ;
    for(int j=0;j<d;j++)
    {
      int n = T[iD+j+1] ;
      int jd = j*d ;
      int nd = n*d ;
      for(int k=0;k<d;k++)
        mat(j,k) = X[nd+k]-X[n0d+k] ;
    }
    v = Mesh::OneOverFact[d]*mat.determinant();
    for(int j=0;j<D;j++)
      volN[T[iD+j]] += v ;
    for(int j=0;j<D;j++)
      nbs[T[iD+j]]++ ;
  }

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
  string fname ;
  for(int i=0;i<argc;i++)
  {
    if(argv[i]==string("-mesh"))
      fname = argv[i+1];
    if(argv[i]==string("-m"))
      M = stol(argv[i+1]);
    if(argv[i]==string("-out"))
      lvlOut = stoi(argv[i+1]);
  } 

  Mesh m(fname);

  if(lvlOut>1)
  {
    m.print();
    //m.print(cout,0);
    if(fname[fname.size()-1]=='t')
      m.write(string(fname)+"b");
  }
  clock_t a,b ;

  double vol = 0.0 ;
  a = clock();
  for(int i=0;i<M;i++)
    vol += m.volume() ;
  b = clock();
  cout << "sum vol= " << vol << " in " << (b-a)/(double)(CLOCKS_PER_SEC) << endl;

  vector<double> volE(m.nbe) ;
  a = clock();
  for(int i=0;i<M;i++)
    m.volumesE(volE);
  b = clock();
  cout << "volumesE (" << volE.size() << " " << volE[0] << ") in " << (b-a)/(double)(CLOCKS_PER_SEC) << endl;
  if(lvlOut>0)
  {
    for(const auto & v : volE)
      cout << v << " ";
    cout << endl;
  }
  vector<double> volN(m.nbn) ;
  a = clock();
  for(int i=0;i<M;i++)
    m.volumesN(volN);
  b = clock();
  cout << "volumesN (" << volN.size() << " " << volN[0] << ") in " << (b-a)/(double)(CLOCKS_PER_SEC) << endl;
  if(lvlOut>0)
  {
    for(const auto & v : volN)
      cout << v << " ";
    cout << endl;
  }
  
  a = clock();
  for(int i=0;i<M;i++)
    m.volumesNavg(volN);
  b = clock();
  cout << "volumesNavg (" << volN.size() << " " << volN[0] << ") in " << (b-a)/(double)(CLOCKS_PER_SEC) << endl;
  if(lvlOut>0)
  {
    for(const auto & v : volN)
      cout << v << " ";
    cout << endl;
  }
  cout << "completed." << endl;
  return 0 ;
}
