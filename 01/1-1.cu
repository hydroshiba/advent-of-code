#include <iostream>
#include <iomanip>
#include <vector>

#include "timer.cuh"

const int THREADS = 1024;

// SOLVING THE PREFIX SUM PROBLEM
/* ############################################################# */

__global__ void block_scan(int* arr, int n, int* aux) {
	__shared__ int mem[THREADS], buffer[THREADS];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int x = threadIdx.x;

	if(i < n) mem[x] = arr[i];
	else mem[x] = 0;
	__syncthreads();

	int* in = mem;
	int* out = buffer;

	for(int stride = 1; stride < blockDim.x; stride *= 2) {
		out[x] = in[x];
		__syncthreads();

		if(x + stride < blockDim.x)
			out[x + stride] += in[x];
		__syncthreads();

		int* temp = in;
		in = out;
		out = temp;
	}

	if(i < n) arr[i] = in[x];
	if(x == blockDim.x - 1) aux[blockIdx.x] = in[x];
}

__global__ void block_adjust(int* arr, int n, int* aux) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	if(blockIdx.x > 0) arr[i] += aux[blockIdx.x - 1];
}

void prefix_sum(int* arr, int n, int* aux) {
	int blocks = (n + THREADS - 1) / THREADS;
	
	block_scan<<<blocks, THREADS>>>(arr, n, aux);
	if(blocks > 1) prefix_sum(aux, blocks, aux + blocks);
	block_adjust<<<blocks, THREADS>>>(arr, n, aux);
}

void prefix_sum(int* arr, int n) {
	int* aux = nullptr;
	int blocks = (n + THREADS - 1) / THREADS;
	cudaMalloc((void**)&aux, (8 * blocks * sizeof(int)) / 7);

	prefix_sum(arr, n, aux);
	cudaDeviceSynchronize();
	cudaFree(aux);
}

// SOLVING THE VALUE COUNTING PROBLEM
/* ############################################################# */

__device__ __inline__ int condition(int val) {
	return !(val % 100);
}

__global__ void count_naive(int* arr, int n, int* count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;

	if(condition(arr[i])) {
		atomicAdd(count, 1);
	}
}

__global__ void count_reduce(int* arr, int n, int* count) {
	__shared__ int mem[THREADS];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int x = threadIdx.x;

	mem[x] = (i < n ? condition(arr[i]) : 0);
	__syncthreads();

	for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
		if(x < stride) mem[x] += mem[x + stride];
		__syncthreads();
	}

	if(x == 0) atomicAdd(count, mem[x]);
}

int value_count(int* arr, int n) {
	int cnt;
	int* dcnt;
	cudaMalloc((void**)&dcnt, sizeof(int));
	cudaMemset(dcnt, 0, sizeof(int));

	int blocks = (n + THREADS - 1) / THREADS;
	count_naive<<<blocks, THREADS>>>(arr, n, dcnt);
	cudaDeviceSynchronize();

	cudaMemcpy(&cnt, dcnt, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dcnt);
	return cnt;
}

int main() {
	std::ios::sync_with_stdio(false);
	std::cin.tie(nullptr);

	std::vector<int> vec = {50};
	char c;

	// Can't parallelize input reading unfortunately
	while(std::cin >> c) {
		int num;
		std::cin >> num;
		vec.push_back(c == 'L' ? -num : num);
	}

	// Copy to device
	int n = vec.size();
	int* data = nullptr;

	cudaMalloc((void**)&data, n * sizeof(int));
	cudaMemcpy(data, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);

	Timer timer;

	timer.start();
	prefix_sum(data, n);
	timer.stop();

	std::cerr << "Prefix sum taken: " << timer.milliseconds() << " ms" <<  std::endl;

	timer.start();
	int ans = value_count(data, n);
	timer.stop();

	std::cerr << "Value count taken: " << timer.milliseconds() << " ms" <<  std::endl;
	
	std::cout << ans << std::endl;
	cudaFree(data);
}