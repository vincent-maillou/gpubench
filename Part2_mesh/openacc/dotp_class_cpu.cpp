#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

class Dotp{
private:
    double *X;
    double *Y;
    double dotp;

    long N;
    int M;

public:
    Dotp(long N_ = 10000, int M_ = 1) : X(new double[N_]), Y(new double[N_]), dotp(0), N(N_), M(M_) {}
    ~Dotp(){
        delete X;
        delete Y;
    }

    void Init(int init_ = 0);
    double Compute();

};

void Dotp::Init(int init_){
    
    switch(init_){
        case 1: // Initialisation sin()/cos()

            for(long i(0); i<N; ++i){
                X[i] = cos(i*2.*M_PI/N);
                Y[i] = sin(i*2.*M_PI/N);
            }
            break;

        default: // Initialisation unitaire 

            for(long i(0); i<N; ++i){
                X[i] = 1.;
                Y[i] = 1.;
                // cout << i << endl;
            }
    }
}

double Dotp::Compute(){

    for(int j(0); j<M; ++j){

        for(long i(0); i<N; ++i){
            dotp += X[i]*Y[i];
        }
    }
    
    return dotp;
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

    for(int i=0;i<argc;i++) // Parsing des options d'execution
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