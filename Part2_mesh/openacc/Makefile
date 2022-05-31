all: dotp_class_cpu.out dotp_class_openacc.out mesh_openacc.out

dotp_class_cpu.out: dotp_class_cpu.cpp
	g++ -O2 dotp_class_cpu.cpp -o dotp_class_cpu.out

dotp_class_openacc.out: dotp_class_openacc.cpp
	nvc++ -O2 -acc dotp_class_openacc.cpp -Minfo=acc -o dotp_class_openacc.out

mesh_openacc.out: mesh_openacc.cpp
	nvc++ -O2 -acc mesh_openacc.cpp -Minfo=acc -o mesh_openacc.out

clean:
	rm -rf *.out

