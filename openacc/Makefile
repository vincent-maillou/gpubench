all: build/init_openacc_managed.out build/init_openacc.out build/dotp_openacc.out

build/init_openacc_managed.out: init_openacc_managed.cpp 
	nvc++ -O2 -acc init_openacc_managed.cpp -Minfo=acc -ta=tesla:managed -o build/init_openacc_managed.out

build/init_openacc.out: init_openacc.cpp 
	nvc++ -O2 -acc init_openacc.cpp -Minfo=acc -ta=tesla -o build/init_openacc.out

build/dotp_openacc.out: dotp_openacc.cpp
	nvc++ -O2 -acc dotp_openacc.cpp -Minfo=acc -ta=tesla -o build/dotp_openacc.out

clean:
	rm -rf build/*.out
