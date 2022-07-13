all: rand_serial.out rand_openacc.out

rand_serial.out: rand_serial.cpp
	g++ -O2 rand_serial.cpp -o rand_serial.out

rand_openacc.out: rand_openacc.cpp
	nvc++ -O2 -acc rand_openacc.cpp -Minfo=acc -o rand_openacc.out

clean:
	rm -rf *.out

