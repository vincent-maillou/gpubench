#include <iostream>
#include <vector>
#include <map>
#include <array>
#include <cmath>
#include <string>
#include <fstream>
using namespace std ;

#include <Eigen/Dense>
using namespace Eigen ;

//#define EMAT Matrix<double,Dynamic,Dynamic> mat(d,d)
#define EMAT Matrix<double,3,3> mat

#include "mpi.h"

#define CHUNK_SIZE 4096

class Hardware
{
public : 
  int rank,size ;
};

Hardware Computer ;

class Vois
{
public :
  Vois() { nbv=0; PosVois.push_back(0) ; } 
  Vois(const Vois & vr,char r); 
  int nbv ;
  vector<int> CoreVois ;
  vector<int> PosVois ;
  vector<pair<int,int>> NodeVois  ;
  void print(ostream & os=cout);
};

void Vois::print(ostream & os)
{
  os << nbv << endl;
  for(const auto & v : CoreVois)
    os << v << " ";
  os << endl;
  for(const auto & v : PosVois)
    os << v << " ";
  os << endl;
  for(const auto & v : NodeVois)
    os << v.first << " " << v.second << " ";
  os << endl;
}

Vois::Vois(const Vois & vr,char r)
{
  vector<int> nbsr(Computer.size,0) ;
  vector<int> nbss(Computer.size,0) ;

  for(int i=0;i<vr.CoreVois.size();i++)
    nbsr[vr.CoreVois[i]]=vr.PosVois[i+1]-vr.PosVois[i] ;
  
  MPI_Alltoall(&(nbsr[0]),1,MPI_INT,&(nbss[0]),1,MPI_INT,MPI_COMM_WORLD);

  PosVois.push_back(0);
  for(int i=0;i<Computer.size;i++)
    if(nbss[i]>0)
    {  
      CoreVois.push_back(i);
      PosVois.push_back(PosVois[PosVois.size()-1]+nbss[i]);
    }
  nbv=CoreVois.size();
  
  vector<MPI_Request> GhostRequests(vr.nbv);
  vector<MPI_Status> GhostStatus(vr.nbv);
  vector<int> GhostNodes(vr.PosVois[vr.nbv]);
  vector<MPI_Request> CloneRequests(nbv);
  vector<MPI_Status> CloneStatus(nbv);
  vector<int> CloneNodes(PosVois[nbv]);
  
  for(int i=0;i<nbv;i++)
    MPI_Irecv(&(CloneNodes[PosVois[i]]),PosVois[i+1]-PosVois[i],MPI_INT,CoreVois[i],1,MPI_COMM_WORLD,&(CloneRequests[i]));

  for(int i=0;i<vr.PosVois[vr.nbv];i++)
    GhostNodes[i] = vr.NodeVois[i].second ;

  for(int i=0;i<vr.nbv;i++)
    MPI_Isend(&(GhostNodes[vr.PosVois[i]]),vr.PosVois[i+1]-vr.PosVois[i],MPI_INT,vr.CoreVois[i],1,MPI_COMM_WORLD,&(GhostRequests[i]));

  MPI_Waitall(nbv,&(CloneRequests[0]),&(CloneStatus[0]));
 
  NodeVois.resize(PosVois[nbv]); 
  for(int i=0;i<nbv;i++)
    for(int j=PosVois[i];j<PosVois[i+1];j++)
    {
      NodeVois[j].first=CoreVois[i] ;
      NodeVois[j].second=CloneNodes[j];
    }
  MPI_Waitall(vr.nbv,&(GhostRequests[0]),&(GhostStatus[0]));
}


template <class T>
void UpdateFromGhosts(int nbn_loc,int d,vector<T> &Field ,const Vois &VC,const Vois &VG,MPI_Datatype MT);

class Mesh
{
public :
  Mesh(const string & fname,const string & ext);

  void print(ostream & os=cout,int decal=1);
  void write(const string & fname);
  void read(const string & fname,const string & ext,map<pair<int,int>,vector<int>> & GhostsPos);

  double volume();
  void volumesE(vector<double> & volE);
  void volumesN(vector<double> & volN);
  void volumesNavg(vector<double> & volN);

  int dom ;
  int d,D ;
  int nbn,nbe ;
  int nbn_loc,nbn_ext;
  vector<double> X ;
  vector<int> T ;

  Vois VGhosts ;
  Vois VClones ;

  static array<double,6> OneOverFact ;
  //static constexpr array<double,6> OneOverFact = {1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0} ;
};
 
array<double,6> Mesh::OneOverFact = {1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0} ;

Mesh::Mesh(const string & fname,const string & ext)
{
  // a) read file  
  map<pair<int,int>,vector<int>> GhostsPos ;
  if(ext == ".T")
  {
    string name=fname+"_"+to_string(Computer.rank)+ext;
    ifstream is(name);
    is >> dom ;
    is >> nbn >> d >> nbe >> D ;
    X.resize(nbn*d);
    for(int i=0;i<X.size();i++)
      is >> X[i] ;
    nbn_loc = nbn ;

    int c,n ;
    T.resize(nbe*D);
    for(int i=0;i<T.size();i++)
    {
      is >> n >> c  ;
      if(c==dom)
      {
        T[i] = n-1 ;
      }
      else
      {
	GhostsPos[pair<int,int>(c,n-1)].push_back(i) ;
      }
    }
  }
  else if(ext==".TB")
   this->read(fname,ext,GhostsPos);

  // b) Build Voisisnage
  /*
    cout << "GhostsPos : " << GhostsPos.size() << " "   << Computer.rank << endl ; 
    for(const auto & v : GhostsPos)
    {
      cout << v.first.first << " " << v.first.second << endl;
      for(const auto & w : v.second)
        cout << w << " ";
      cout << endl;
    }
*/
    VGhosts.NodeVois.resize(GhostsPos.size());
    nbn_ext = 0 ;
    for(const auto & v : GhostsPos)
    {
      VGhosts.NodeVois[nbn_ext]=v.first;
      for(const auto & w : v.second)
        T[w] = nbn_loc + nbn_ext ; 
      nbn_ext++;
    }
    nbn=nbn_loc+nbn_ext;
    //cout << "rank " << Computer.rank << " nbn " << nbn << " " << nbn_loc << " " << nbn_ext << endl; 
    int prev = -1 ;
    int nb = 0 ;
    VGhosts.nbv=0;
    VGhosts.PosVois.resize(0);
    for(int i=0;i<VGhosts.NodeVois.size();i++)
    {
      int c = VGhosts.NodeVois[i].first ;
      if(c!=prev)
      {
        VGhosts.PosVois.push_back(nb);
	VGhosts.CoreVois.push_back(c);
	VGhosts.nbv++;
        prev = c ; 
      }
      nb++;
    }
    VGhosts.PosVois.push_back(nb);
    //cout << "rank= " << Computer.rank << " VGhosts" <<  endl;
    //VGhosts.print();

    VClones=Vois(VGhosts,'a');
    //cout << "rank= " << Computer.rank << " VClones" << endl;
    //VClones.print();

  // c) Update XGhost  
  X.resize(nbn*d);
  int nbc = VClones.PosVois[VClones.nbv] ;
  int nbg = VGhosts.PosVois[VGhosts.nbv] ;
  vector<double> XSend(nbc*d);
  //vector<double> XRecv(nbg*d);

  vector<MPI_Request> GhostRequests(VGhosts.nbv);
  vector<MPI_Status> GhostStatus(VGhosts.nbv);
  vector<MPI_Request> CloneRequests(VClones.nbv);
  vector<MPI_Status> CloneStatus(VClones.nbv);
  
  for(int i=0;i<VGhosts.nbv;i++)
    MPI_Irecv(&(X[(nbn_loc+VGhosts.PosVois[i])*d]),(VGhosts.PosVois[i+1]-VGhosts.PosVois[i])*d,MPI_DOUBLE,VGhosts.CoreVois[i],2,MPI_COMM_WORLD,&(GhostRequests[i]));
    //MPI_Irecv(&(XRecv[VGhosts.PosVois[i]*d]),(VGhosts.PosVois[i+1]-VGhosts.PosVois[i])*d,MPI_DOUBLE,VGhosts.CoreVois[i],2,MPI_COMM_WORLD,&(GhostRequests[i]));

  for(int i=0;i<nbc;i++)
    for(int j=0;j<d;j++)
      XSend[i*d+j]=X[VClones.NodeVois[i].second*d+j];
  
  /*
  for(const auto & v : XSend)
    cout << v << " ";
  cout << " XSend " <<  Computer.rank << endl;
*/

  for(int i=0;i<VClones.nbv;i++)
    MPI_Isend(&(XSend[VClones.PosVois[i]*d]),(VClones.PosVois[i+1]-VClones.PosVois[i])*d,MPI_DOUBLE,VClones.CoreVois[i],2,MPI_COMM_WORLD,&(CloneRequests[i]));

  MPI_Waitall(VGhosts.nbv,&(GhostRequests[0]),&(GhostStatus[0]));
 
  //for(const auto & v : XRecv)
  //  cout << v << " ";
  //cout << " XRecv " << Computer.rank << endl;
  //for(const auto & v : X)
  //  cout << v << " ";
  //cout << " X " <<  Computer.rank << endl;

/*  
  ofstream ofs("fff_"+std::to_string(Computer.rank)+".t");
  this->print(ofs);
  */
  MPI_Waitall(VClones.nbv,&(CloneRequests[0]),&(CloneStatus[0]));
}

void Mesh::print(ostream & os,int decal)
{
  os << nbn << " " << d << " " << nbe << " " << D << endl;
  for(int i=0;i<nbn;i++)
  {
    for(int j=0;j<d;j++)
      os << X[i*d+j] << " ";
    os << endl;
  }
  for(int i=0;i<nbe;i++)
  {
    for(int j=0;j<D;j++)
      os << T[i*D+j]+decal << " ";
    os << endl;
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

void Mesh::read(const string & fname,const string & ext,map<pair<int,int>,vector<int>> & GhostsPos)
{
#ifdef CHUNK_SIZE
  char buf[256] ;
  string filename = fname ;
  filename += "/Chunk-" ;
  //filename += std::to_string(static_cast<long long>((rank/FloorSize+0)*FloorSize-0)) ;
  sprintf(buf,"%07d",(Computer.rank/CHUNK_SIZE+0)*CHUNK_SIZE-0) ;
  filename += buf ;
  filename += "_" ;
  //filename += std::to_string(static_cast<long long>((rank/FloorSize+1)*FloorSize-1)) ;
  sprintf(buf,"%07d",(Computer.rank/CHUNK_SIZE+1)*CHUNK_SIZE-1) ;
  filename += buf ;
  filename += "/" ;

  filename += fname.substr(fname.find_last_of("/") + 1) ;
  filename += "_" ;
  filename += std::to_string(static_cast<long long>(Computer.rank)) ;
  filename += ".TB" ;
#else
  string filename ;
  {
    ostringstream oss;
    oss << fname  << "_" << Computer.rank << ".TB" ;
    filename = oss.str();
  }
#endif
 
  ifstream is(filename);

  if(!is.is_open())
    cerr << "Error the file " << filename << " do not exist" << endl; 
  else
  {
    is.read((char*)&dom,sizeof(int)) ;
    array<int,4> data ;
    is.read((char*)&(data[0]),4*sizeof(int)) ;
    d = data[0];
    D = data[1];
    nbn_loc = data[2];
    nbe = data[3] ;

    int size ;
    size = d * nbn_loc ;
    X.resize(size) ;
    is.read((char*)&(X[0]),size*sizeof(double)) ;

    size = nbe * D ;
    T.resize(size) ;

    int pos = 0 ;
    int nb, rest ;
    int BufferSize = 100000 ;

    nb = nbe / BufferSize ;
    rest = nbe % BufferSize ;

    vector<int> TT(2*BufferSize*D) ;

    auto func = [this,&pos,&TT,&is,&GhostsPos](int n,int bs,int s) -> void
    {
      is.read((char*)&(TT[0]),2*bs*D*sizeof(int)) ;
      /*
      for(const auto & v : TT)
        cout << v << " " ;
      cout << endl;
      */

      for(int i=0;i<s;i++)
      {
        int ib = i*D;
        int ab = n*bs*D + ib ;
        int ib2 = 2*ib;
        for(int j=0;j<D;j++)
        {
          if(TT[ib2+2*j+1]==dom)
          {
            T[pos] = TT[ib2+2*j+0]-1;
          }
          else
          {
	    GhostsPos[pair<int,int>(TT[ib2+2*j+1],TT[ib2+2*j+0]-1)].push_back(pos) ;
//	    cout << "exttttttttttttt " << GhostsPos.size() << endl;
          }
          pos++ ;
        }
      }
    } ;

    for(int n=0;n<nb;n++)
      func(n,BufferSize,BufferSize) ;

    if(rest)
      func(nb,BufferSize,rest) ;
  }
  //cout << "GhostsPos in read : " << GhostsPos.size() << " "   << Computer.rank << endl ; 
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
  double rvol;
  MPI_Allreduce(&vol,&rvol,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  vol = rvol ;
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
  UpdateFromGhosts<double>(nbn_loc,1,volN,VClones,VGhosts,MPI_DOUBLE);
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
  UpdateFromGhosts<double>(nbn_loc,1,volN,VClones,VGhosts,MPI_DOUBLE);
  UpdateFromGhosts<int>(nbn_loc,1,nbs,VClones,VGhosts,MPI_INT);

  for(int i=0;i<nbn;i++)
    volN[i] /= nbs[i] ;
}

template <class T>
void UpdateFromGhosts(int nbn_loc,int d,vector<T> &Field ,const Vois &VC,const Vois &VG,MPI_Datatype MT)
{
  int nbg = VG.PosVois[VG.nbv] ;
  int nbc = VC.PosVois[VC.nbv] ;
  //vector<T> bufS(nbg);
  vector<T> bufR(nbc);
  
  vector<MPI_Request> GRequests(VG.nbv);
  vector<MPI_Status> GStatus(VG.nbv);
  vector<MPI_Request> CRequests(VC.nbv);
  vector<MPI_Status> CStatus(VC.nbv);
  
  for(int i=0;i<VC.nbv;i++)
    MPI_Irecv(&(bufR[VC.PosVois[i]*d]),(VC.PosVois[i+1]-VC.PosVois[i])*d,MT,VC.CoreVois[i],3,MPI_COMM_WORLD,&(CRequests[i]));

  for(int i=0;i<VG.nbv;i++)
    MPI_Isend(&(Field[(nbn_loc+VG.PosVois[i])*d]),(VG.PosVois[i+1]-VG.PosVois[i])*d,MT,VG.CoreVois[i],3,MPI_COMM_WORLD,&(GRequests[i]));

  MPI_Waitall(VC.nbv,&(CRequests[0]),&(CStatus[0]));
/* 
  for(const auto & v : bufR)
    cout << v << " ";
  cout << " bufR " << Computer.rank << endl;
*/
  for(int i=0;i<nbc;i++)
    for(int j=0;j<d;j++)
      Field[VC.NodeVois[i].second*d+j] += bufR[i*d+j];

  MPI_Waitall(VC.nbv,&(CRequests[0]),&(CStatus[0]));
}


int main(int argc,char**argv)
{
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&Computer.rank);
  MPI_Comm_size(MPI_COMM_WORLD,&Computer.size);

  if(Computer.rank==0)
  {
    for(int i=0;i<argc;i++)
      cout << argv[i] << " " ;
    cout << endl;
  }

  int lvlOut = 0 ;
  int M = 1 ;
  string fname ;
  string ext ;
  for(int i=0;i<argc;i++)
  {
    if(argv[i]==string("-mesh"))
    {
      fname = argv[i+1];
      ext = argv[i+2];
    }
    if(argv[i]==string("-m"))
      M = stol(argv[i+1]);
    if(argv[i]==string("-out"))
      lvlOut = stoi(argv[i+1]);
  } 

  Mesh m(fname,ext);

  if(lvlOut>1)
  {
    m.print();
    //m.print(cout,0);
    if(fname[fname.size()-1]=='t')
      m.write(string(fname)+"b");
  }
  double a,b ;
  double vol = 0.0 ;
  MPI_Barrier(MPI_COMM_WORLD);
  a = MPI_Wtime();
  for(int i=0;i<M;i++)
    vol += m.volume() ;
  MPI_Barrier(MPI_COMM_WORLD);
  b = MPI_Wtime();
  if(Computer.rank==0)
    cout << "sum vol= " << vol << " in " << b-a << "s" << endl;

  vector<double> volE(m.nbe) ;
  MPI_Barrier(MPI_COMM_WORLD);
  a = MPI_Wtime();
  for(int i=0;i<M;i++)
    m.volumesE(volE);
  MPI_Barrier(MPI_COMM_WORLD);
  b = MPI_Wtime();
  if(Computer.rank==0)
    cout << "volumesE (" << volE.size() << " " << volE[0] << ") in " << b-a << "s"  << endl;
  if(lvlOut>0)
  {
    for(const auto & v : volE)
      cout << v << " ";
    cout << endl;
  }
  vector<double> volN(m.nbn) ;
  MPI_Barrier(MPI_COMM_WORLD);
  a = MPI_Wtime();
  for(int i=0;i<M;i++)
    m.volumesN(volN);
  MPI_Barrier(MPI_COMM_WORLD);
  b = MPI_Wtime();
  if(Computer.rank==0)
    cout << "volumesN (" << volN.size() << " " << volN[0] << ") in " << b-a << "s" << endl;
  if(lvlOut>0)
  {
    for(const auto & v : volN)
      cout << v << " ";
    cout << endl;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  a = MPI_Wtime();
  for(int i=0;i<M;i++)
    m.volumesNavg(volN);
  MPI_Barrier(MPI_COMM_WORLD);
  b = MPI_Wtime();
  if(Computer.rank==0)
    cout << "volumesNavg (" << volN.size() << " " << volN[0] << ") in " << b-a << "s" << endl;
  if(lvlOut>0)
  {
    for(const auto & v : volN)
      cout << v << " ";
    cout << endl;
  }

  if(Computer.rank==0)
    cout << "completed." << endl;

  MPI_Finalize();
  return 0 ;
}



