#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <fstream>
#include <ctime>
#include <chrono>

#include "mesh_functors.hpp"

using namespace std ;

/* TODO
  - Implémenter volume [x]
  - Vérifier exactitude volume 2D []
  - Vérifier exactitude volume 3D [x]

  - Implémenter volN []
  - Vérifier exactitude volE 2D []
  - Vérifier exactitude volE 3D []

  - Implémenter volE []
  - Vérifier exactitude volE 2D []
  - Vérifier exactitude volE 3D []

  - Implémenter volNAverage []
  - Vérifier exactitude volNAverage 2D []
  - Vérifier exactitude volNAverage 3D []  
*/

class Mesh
{
public :
  Mesh(const string & fname);
  ~Mesh();

  void print(ostream & os=cout,int decal=1);
  void write(const string & fname);
  void read(const string & fname);

  double volume();

  double determinant(double mat) const;
  double determinant(double mat[2][2]) const;
  double determinant(double mat[3][3]) const;

  int d,D; 
  int nbn,nbe;
  vector<double> X;
  vector<int> T;

  // Kokkos dualview array
  view_type_double X_dv;
  view_type_integer T_dv;

  double volume_sum;

  const array<double,6> OneOverFact;
};

Mesh::Mesh(const string & fname) : OneOverFact({1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0})
{
    auto ext = fname.substr(fname.find_last_of("."));
    if(ext == ".t") // Si .t parsing des arguments
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
    else if(ext==".tb") // Si .tb (binaire) pas de parsing
    this->read(fname);

    /* Resize and filling of dualviews vectors */
    Kokkos::resize(X_dv, nbn*d);
    Kokkos::resize(T_dv, nbe*D);

    for (view_type_double::size_type i(0); i < X_dv.extent(0); ++i) {
        X_dv.h_view(i) = X[i];
    }
    X_dv.modify<view_type_double::host_mirror_space> ();

    for (view_type_integer::size_type i(0); i < T_dv.extent(0); ++i) {
        T_dv.h_view(i) = T[i];
    }
    T_dv.modify<view_type_integer::host_mirror_space> ();

    volume_sum = 0.;
}

Mesh::~Mesh(){
  ;
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
  Kokkos::parallel_reduce(nbe, Volume<view_type_double::execution_space>(X_dv, T_dv, d, D, OneOverFact[d]), volume_sum);
  Kokkos::fence();

  return volume_sum;
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


  Kokkos::initialize (argc, argv);
  {
    Mesh m(fname);

    /* if(lvlOut>1)
    {
      m.print();
      
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

    cout << endl << "completed." << endl;
  }
  Kokkos::finalize();

  return 0 ;
}
