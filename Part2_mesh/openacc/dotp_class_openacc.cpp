#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

class Dotp{
private:
    double *restrict X;
    double *restrict Y;

    long const N;
    int const M;

public:
    Dotp(long N_ = 10000, int M_ = 1) : X(new double[N_]), Y(new double[N_]), N(N_), M(M_) 
    {
        /* Need to use: unstructured data lifetimes
            - use of "data" word to identifie strat and end of data lifetime
            - with "data" only: create and copyin works
            - the "exit data" directive accepts: copyout and delete data clauses
        */
        #pragma acc enter data copyin(this)
        #pragma acc enter data create(X[:N], Y[:N])
    }

    ~Dotp(){
        #pragma acc exit data delete(this) 
        #pragma acc exit data delete(X[:N], Y[:N])
        
        delete [] X;
        delete [] Y;
    }

    void Init(int init_ = 0);
    double Compute();

};

void Dotp::Init(int init_){

    switch(init_){
        case 1: // Initialisation sin()/cos()

            #pragma acc parallel loop present(X[N], Y[N])
            for(long i(0); i<N; ++i){
                X[i] = cos(i*2.*M_PI/N);
                Y[i] = sin(i*2.*M_PI/N);
            }
            break;

        default: // Initialisation unitaire 

            #pragma acc parallel loop present(X[N], Y[N])
            for(long i(0); i<N; ++i){
                X[i] = 1.;
                Y[i] = 1.;
            }
    }
}

double Dotp::Compute(){

    double dotp_loc(0);

    #pragma acc data present(X[:N], Y[:N])
    #pragma acc parallel loop reduction(+:dotp_loc)
    for(long i(0); i<N; ++i){
        dotp_loc += X[i]*Y[i];
    }
    return dotp_loc;
}



int main(int argc,char**argv)
{
    for(int i=0;i<argc;i++){
        cout << argv[i] << " " ;
    }
    cout << endl;

    long N(10000);
    int M(1);
    int Init(1);

    for(int i=0;i<argc;i++)
    {
        if(argv[i]==string("-n"))
            N = stol(argv[i+1]);
        if(argv[i]==string("-m"))
            M = stol(argv[i+1]);
        if(argv[i]==string("-init"))
            Init = stol(argv[i+1]);
    }     

    Dotp dotp(N, M);
    dotp.Init(Init);

    auto begin = std::chrono::high_resolution_clock::now();  

        double dotp_sum(dotp.Compute());

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  

    cout << "Dot-product computation of size " << N << ", result in the following summation: " << dotp_sum << endl;
    cout << "   time: " << elapsed.count()/1000.0 << endl;

    return 0;
}
