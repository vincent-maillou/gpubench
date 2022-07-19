all: mesh_serial.out mesh_omp.out

mesh_serial.out: mesh_serial.cpp
	g++ -O2 mesh_serial.cpp -o mesh_serial.out

mesh_omp.out: mesh_omp.cpp
	g++ -O2 -fopenmp mesh_omp.cpp -o mesh_omp.out

clean:
	rm -rf *.out

