all: init_cuda.out dotp_cuda.out get_properties.out

init_cuda.out: init_cuda.cu 
	nvcc -O2 init_cuda.cu -o build/init_cuda.out

dotp_cuda.out: dotp_cuda.cu 
	nvcc -O2 -I/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/math_libs/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/math_libs/lib64 -lcublas dotp_cuda.cu -o build/dotp_cuda.out

get_properties.out: get_properties.cu
	nvcc get_properties.cu -o build/get_properties.out

clean:
	rm -rf build/*.out
