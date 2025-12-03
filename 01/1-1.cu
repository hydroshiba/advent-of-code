#include <iostream>
#include <iomanip>
#include <vector>

#include "timer.cuh"

const int THREADS = 128;

// CUDA KERNELS
/* ############################################################# */

__device__ __inline__ int condition(int val) {
	return !(val % 100);
}

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

// ACTUALLY SOLVING THE PUZZLE
/* ############################################################# */

void prefix_sum(int* arr, int n, int* aux) {
	int blocks = (n + THREADS - 1) / THREADS;
	
	block_scan<<<blocks, THREADS>>>(arr, n, aux);
	if(blocks > 1) prefix_sum(aux, blocks, aux + blocks);
	block_adjust<<<blocks, THREADS>>>(arr, n, aux);
}

int solve(int* arr, int n) {
	Timer timer;
	int ans, blocks = (n + THREADS - 1) / THREADS;
	int* ans_dev = nullptr;
	int* aux = nullptr;

	cudaMalloc((void**)&ans_dev, sizeof(int)); cudaMemset(ans_dev, 0, sizeof(int));
	cudaMalloc((void**)&aux, 2 * blocks * sizeof(int));

	timer.start();
	prefix_sum(arr, n, aux);
	timer.stop();
	std::cerr << "Prefix sum time: " << timer.milliseconds() << " ms\n";

	timer.start();
	// Naive kernel seems to do better because of low count rate
	count_naive<<<blocks, THREADS>>>(arr, n, ans_dev);
	timer.stop();
	std::cerr << "Count time: " << timer.milliseconds() << " ms\n";

	cudaMemcpy(&ans, ans_dev, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(ans_dev);
	cudaFree(aux);

	return ans;
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

	int n = vec.size();
	int* data = nullptr;

	cudaMalloc((void**)&data, n * sizeof(int));
	cudaMemcpy(data, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);
	
	int ans = solve(data, n);
	cudaFree(data);
	std::cout << ans;
}