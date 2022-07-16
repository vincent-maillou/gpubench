all: rand_serial.out rand_openacc.out rand_openacc_2.out

rand_serial.out: rand_serial.cpp
	g++ -O2 rand_serial.cpp -o rand_serial.out

rand_openacc.out: rand_openacc.cpp
	nvc++ -O2 -acc=gpu -cudalib=curand -Minfo=acc  rand_openacc.cpp -o rand_openacc.out

rand_openacc_2.out: rand_openacc_2.cpp
	nvc++ -O2 -acc=gpu -cudalib=curand -Minfo=acc  rand_openacc_2.cpp -o rand_openacc_2.out

clean:
	rm -rf *.out

