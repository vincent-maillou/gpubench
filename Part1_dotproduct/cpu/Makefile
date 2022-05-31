all: init_serial.out init_omp.out dotp_serial.out dotp_omp.out

init_serial.out: init_serial.cpp
	g++ -O2 -o ./build/init_serial.out init_serial.cpp

init_omp.out: init_omp.cpp
	g++ -O2 -fopenmp -o ./build/init_omp.out init_omp.cpp

dotp_serial.out: dotp_serial.cpp 
	g++ -O2 -o ./build/dotp_serial.out dotp_serial.cpp 

dotp_omp.out: dotp_omp.cpp
	g++ -O2 -fopenmp -o ./build/dotp_omp.out dotp_omp.cpp

clean:
	rm -rf ./build/*.out
