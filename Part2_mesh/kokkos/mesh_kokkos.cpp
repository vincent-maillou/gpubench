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
  - Vérifier exactitude volume 2D [x]
  - Vérifier exactitude volume 3D [x]

  - Implémenter volN [x]
  - Vérifier exactitude volE 2D []
  - Vérifier exactitude volE 3D [x]

  - Implémenter volE [x]
  - Vérifier exactitude volE 2D [x]
  - Vérifier exactitude volE 3D [x]

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
  void volumesE(view_type_double volE);
  void volumesN(view_type_double volN);
  void volumesNavg(view_type_double volNavg);

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
                  MESH::volumesE()
   ------------------------------------------- */

void Mesh::volumesE(view_type_double volE)
{
  Kokkos::parallel_for(nbe, VolumesE<view_type_double::execution_space>(X_dv, T_dv, volE, d, D, OneOverFact[d]));
  Kokkos::fence();
}



/* -------------------------------------------
                  MESH::volumesN()
   ------------------------------------------- */

void Mesh::volumesN(view_type_double volN)
{
  Kokkos::parallel_for(nbe, VolumesN<view_type_double::execution_space>(X_dv, T_dv, volN, d, D, OneOverFact[d]));
  Kokkos::fence();
}



/* -------------------------------------------
                  MESH::volumesNavg()
   ------------------------------------------- */

void Mesh::volumesNavg(view_type_double volNavg)
{
  view_type_integer pnbs ("pnbs vector", nbn); 

  Kokkos::parallel_for(nbe, VolumesNavg<view_type_double::execution_space>(X_dv, T_dv, volNavg, pnbs, d, D, OneOverFact[d]));
  Kokkos::parallel_for(nbn, Avg<view_type_double::execution_space>(volNavg, pnbs));
  Kokkos::fence();
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



    /* ---------computation of volumesE()--------- */

    view_type_double volE ("volE vector", m.nbe); 

    begin = std::chrono::high_resolution_clock::now();  

      for(int i=0;i<M;i++)
        m.volumesE(volE) ;

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << endl << "Mesh volumesE() computation:" << endl;
    cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

    if(lvlOut >= 1){
      volE.sync<view_type_double::host_mirror_space> ();
      
      switch(lvlOut){
        case 1:
          cout << "   data: ";
          for(view_type_double::size_type i(0); i < volE.extent(0); ++i){
            cout << volE.h_view(i) << " ";
          }
          cout << endl;
          break;
        case 2:
          ofstream ofs("volE_kokkos.txt");
          for(view_type_double::size_type i(0); i < volE.extent(0); ++i){
            ofs << volE.h_view(i) << std::endl;
          }
          break;
      } 
    }



    /* ---------computation of volumesN()--------- */

    view_type_double volN ("volN vector", m.nbn); 

    begin = std::chrono::high_resolution_clock::now();  

      for(int i=0;i<M;i++)
        m.volumesN(volN) ;

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << endl << "Mesh volumesN() computation:" << endl;
    cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

    if(lvlOut >= 1){
      volN.sync<view_type_double::host_mirror_space> ();
      
      switch(lvlOut){
        case 1:
          cout << "   data: ";
          for(view_type_double::size_type i(0); i < volN.extent(0); ++i){
            cout << volN.h_view(i) << " ";
          }
          cout << endl;
          break;
        case 2:
          ofstream ofs("volN_kokkos.txt");
          for(view_type_double::size_type i(0); i < volN.extent(0); ++i){
            ofs << volN.h_view(i) << std::endl;
          }
          break;
      } 
    }



    /* ---------computation of volumesNavg()--------- */

    view_type_double volNavg ("volNavg vector", m.nbn); 

    begin = std::chrono::high_resolution_clock::now();  

      for(int i=0;i<M;i++)
        m.volumesNavg(volNavg) ;

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << endl << "Mesh volumesNavg() computation:" << endl;
    cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

    if(lvlOut >= 1){
      volNavg.sync<view_type_double::host_mirror_space> ();
      
      switch(lvlOut){
        case 1:
          cout << "   data: ";
          for(view_type_double::size_type i(0); i < volNavg.extent(0); ++i){
            cout << volNavg.h_view(i) << " ";
          }
          cout << endl;
          break;
        case 2:
          ofstream ofs("volNavg_kokkos.txt");
          for(view_type_double::size_type i(0); i < volNavg.extent(0); ++i){
            ofs << volNavg.h_view(i) << std::endl;
          }
          break;
      } 
    }


    cout << endl << "completed." << endl;
  }
  Kokkos::finalize();

  return 0 ;
}
