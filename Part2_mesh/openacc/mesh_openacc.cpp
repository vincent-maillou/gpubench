#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <fstream>
#include <ctime>
#include <chrono>

using namespace std ;

#define EMAT double mat[3][3] // Cas 3D
// #define EMAT double mat[2][2]

/* TODO
  - Implémenter volN [x]
  - Vérifier volN pas cassé 2D [x]
  - Vérifier volN pas cassé 3D [x]

  - Implémenter volE [x]
  - Vérifier exactitude volE 2D [x]
  - Vérifier exactitude volE 3D [x]

  - Implémenter volNAverage [x]
  - Vérifier exactitude volNAverage 2D [x]
  - Vérifier exactitude volNAverage 3D [x]  

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
  void volumesE(double* , size_t);
  void volumesN(double* , size_t);
  void volumesNavg(double* , size_t);

  double determinant(double mat) const;
  double determinant(double mat[2][2]) const;
  double determinant(double mat[3][3]) const;

  int d,D; 
  int nbn,nbe;
  vector<double> X;
  vector<int> T;

  // Pointeurs sur nos tableaux CPU pout utilisation GPU
  double *restrict pX; 
  size_t tX;
  int *restrict pT; 
  size_t tT;

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

  pX = &X[0]; 
  tX = X.size();
  pT = &T[0]; 
  tT = T.size();

  /* Making the copy of the entire mesh entity from the cpu to the gpu */
  #pragma acc enter data copyin(this)
  #pragma acc enter data copyin(pT[:tT], pX[:tX])
}

Mesh::~Mesh(){
  /* Deleting the mesh entity on the gpu */
  #pragma acc exit data delete(this)
  #pragma acc exit data delete(pT[:tT], pX[:tX])
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

double Mesh::volume()  // Safe to paralelize such full independency is achieve here
{
    
  double vol(0.0); 

  #pragma acc data present(pT[:tT], pX[:tX])
  #pragma acc parallel loop reduction(+:vol)
  for(int i(0); i<nbe; ++i){

    EMAT;

    int iD = i*D ;
    int n0 = pT[iD] ;
    int n0d = n0*d ;

    #pragma acc loop seq
    for(int j(0); j<d; ++j)
    {
      int n = pT[iD+j+1] ;
      int nd = n*d ;

      #pragma acc loop seq
      for(int k(0); k<d; ++k){
        mat[j][k] = pX[nd+k]-pX[n0d+k];
      }
    }

    vol += determinant(mat);
  }

  return vol ;
}



/* -------------------------------------------
                  MESH::volumesE
   ------------------------------------------- */

void Mesh::volumesE(double* pvolE, size_t tvolE)
{
  
  #pragma acc data present(pT[:tT], pX[:tX], pvolE[:tvolE])
  #pragma acc parallel loop
  for(int i=0;i<nbe;i++)
  {

    double v ;
    EMAT ;

    int iD = i*D ;
    int n0 = pT[iD] ;
    int n0d = n0*d ;

    #pragma acc loop seq
    for(int j(0); j<d; ++j)
    {
      int n = pT[iD+j+1];
      int nd = n*d;

      #pragma acc loop seq
      for(int k(0); k<d; ++k)
        mat[j][k] = pX[nd+k]-pX[n0d+k];
    }

    v = determinant(mat);
    pvolE[i] = v;
  }
}



/* -------------------------------------------
                  MESH::volumesN()
   ------------------------------------------- */

void Mesh::volumesN(double* pvolN, size_t tvolN)
{

  /* Initialisation du vecteur sur le GPU */
  #pragma acc data present(pvolN[:tvolN])
  #pragma acc parallel loop
  for(int i(0); i<nbn; ++i){
    pvolN[i] = 0.;
  }

  #pragma acc data present(pT[:tT], pX[:tX], pvolN[:tvolN])
  #pragma acc parallel loop
  for(int i=0;i<nbe;i++)
  {

    double v(0.0);
    EMAT ;

    int iD = i*D ;
    int n0 = pT[iD] ;
    int n0d = n0*d ;

    #pragma acc loop seq
    for(int j=0;j<d;j++)
    {
      int n = pT[iD+j+1] ;
      int nd = n*d ;

      #pragma acc loop seq
      for(int k=0;k<d;k++){
          mat[j][k] = pX[nd+k]-pX[n0d+k];
      }
    }

    v = determinant(mat);
    
    #pragma acc loop seq
    for(int j=0;j<D;j++){
      #pragma acc atomic update
      pvolN[pT[iD+j]] += v;
    }
  }
}



/* -------------------------------------------
              MESH::volumesNavg()
   ------------------------------------------- */

void Mesh::volumesNavg(double* pvolNavg, size_t tvolNavg)
{
  vector<int> nbs(nbn);

  int* pnbs(&nbs[0]);
  size_t tnbs(nbs.size());

  // Fill the nbs vector with "0"
  #pragma acc enter data create(pnbs[:tnbs])
  #pragma acc parallel loop
  for(int i(0); i<nbn; ++i){
    pnbs[i] = 0.;
  }

  /* Initialisation du vecteur sur le GPU */
  #pragma acc data present(pvolNavg[:tvolNavg])
  #pragma acc parallel loop
  for(int i(0); i<nbn; ++i){
    pvolNavg[i] = 0.;
  }

  #pragma acc data present(pT[:tT], pX[:tX], pvolNavg[:tvolNavg], pnbs[:tnbs])
  #pragma acc parallel loop
  for(int i=0;i<nbe;i++)
  {

    double v(0.);
    EMAT ;

    int iD = i*D ;
    int n0 = pT[iD] ;
    int n0d = n0*d ;

    #pragma acc loop seq
    for(int j=0;j<d;j++)
    {
      int n = pT[iD+j+1] ;
      int nd = n*d ;

      #pragma acc loop seq
      for(int k=0;k<d;k++){
        mat[j][k] = pX[nd+k]-pX[n0d+k];
      }
    }

    v = determinant(mat);

    #pragma acc loop seq
    for(int j(0); j<D; ++j){
      #pragma acc atomic update
      pvolNavg[pT[iD+j]] += v ;
    }

    #pragma acc loop seq
    for(int j(0); j<D; ++j){
      #pragma acc atomic update
      pnbs[pT[iD+j]]++ ;
    }
  }

  #pragma acc data present(pvolNavg[:tvolNavg], pnbs[:tnbs])
  #pragma acc parallel loop
  for(int i(0); i<nbn; ++i){
    pvolNavg[i] = pvolNavg[i]/pnbs[i] ;
  }

  #pragma acc exit data delete(pnbs[:tnbs])
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

  vector<double> volE(m.nbe);

  double* pvolE(&volE[0]);
  size_t tvolE(volE.size());

  #pragma acc enter data create(pvolE[:tvolE])


    begin = std::chrono::high_resolution_clock::now();  

    for(int i=0;i<M;i++){
      m.volumesE(pvolE, tvolE);
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << endl << "Mesh volumesE() computation:" << endl;
    cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

    #pragma acc update self(pvolE[:tvolE])

    if(lvlOut >= 1){
      #pragma acc update self(pvolE[:tvolE])
      switch(lvlOut){
        case 1:
          cout << "   data: ";
          for(auto elem : volE){
            cout << elem << " ";
          }
          cout << endl;
          break;
        case 2:
          ofstream ofs("volE_acc.txt");
          for(auto elem : volE){
            ofs << elem << endl;
          }
          break;
      } 
    }


  #pragma acc exit data delete(pvolE[:tvolE])



  /* ---------computation of volumesN()--------- */

  vector<double> volN(m.nbn);

  double* pvolN(&volN[0]);
  size_t tvolN(volN.size());

  #pragma acc enter data create(pvolN[:tvolN])


    begin = std::chrono::high_resolution_clock::now();  

    for(int i=0;i<M;i++){
      m.volumesN(pvolN, tvolN);
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << endl << "Mesh volumesN() computation:" << endl;
    cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

    if(lvlOut >= 1){
      #pragma acc update self(pvolN[:tvolN])
      switch(lvlOut){
        case 1:
          cout << "   data: ";
          for(auto elem : volN){
            cout << elem << " ";
          }
          cout << endl;
          break;
        case 2:
          ofstream ofs("volN_acc.txt");
          for(auto elem : volN){
            ofs << elem << endl;
          } 
        break;
      }  
    }


  #pragma acc exit data delete(pvolN[:tvolN])



  /* ---------computation of volumesNavg()--------- */

  vector<double> volNavg(m.nbn);

  double* pvolNavg(&volNavg[0]);
  size_t tvolNavg(volNavg.size());

  #pragma acc enter data create(pvolNavg[:tvolNavg])


    begin = std::chrono::high_resolution_clock::now();  

    for(int i=0;i<M;i++){
      m.volumesNavg(pvolNavg, tvolNavg);
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << endl << "Mesh volumesNavg() computation: " << endl;
    cout << "   time: " << elapsed.count()/1000000000.0 << " [s]" << endl;

    if(lvlOut >= 1){
      #pragma acc update self(pvolNavg[:tvolNavg])
      switch(lvlOut){
        case 1:
          cout << "   data: ";
          for(auto elem : volNavg){
            cout << elem << " ";
          }
          cout << endl;
          break;
        case 2:
          ofstream ofs("volNavg_acc.txt");
          for(auto elem : volNavg){
            ofs << elem << endl;
          } 
        break;
      }  
    }


  #pragma acc exit data delete(pvolNavg[:tvolNavg])


  cout << endl << "completed." << endl;
  return 0 ;
}
