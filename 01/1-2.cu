#include <iostream>
#include <iomanip>
#include <vector>

#include "timer.cuh"

const int THREADS = 128;

// CUDA KERNELS
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

__global__ void transform_divmod(int* arr, int n, int* res, int div) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n) {
		res[i] = abs(arr[i]) / div;
		arr[i] %= div;
	}
}

__global__ void transform_reduce(int* arr, int n, int* res, int div, int* count) {
	__shared__ int mem[THREADS];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int x = threadIdx.x;

	// Load & transform

	if(i < n) mem[x] = res[i]; 
	else mem[x] = 0;

	if(i + 1 < n) {
		int l = arr[i], r = arr[i + 1], cur = l;
		if(l > r) l ^= r ^= l ^= r; // dirty swap
		if(r < 0) r -= div - 1;
		
		int val = (r / div) * div;
		mem[x] += (cur != val && l <= val);
	}

	__syncthreads();

	// Reduction on shared memory

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
	
	int* cnt = nullptr;
	int* ans_dev = nullptr;
	int* aux = nullptr;

	cudaMalloc((void**)&cnt, n * sizeof(int)); cudaMemset(cnt, 0, n * sizeof(int));
	cudaMalloc((void**)&ans_dev, sizeof(int)); cudaMemset(ans_dev, 0, sizeof(int));
	cudaMalloc((void**)&aux, 2 * blocks * sizeof(int));

	timer.start();
	transform_divmod<<<blocks, THREADS>>>(arr, n, cnt, 100);
	timer.stop();
	std::cerr << "Divmod transform time: " << timer.milliseconds() << " ms\n";

	timer.start();
	prefix_sum(arr, n, aux);
	timer.stop();
	std::cerr << "Prefix sum time: " << timer.milliseconds() << " ms\n";

	timer.start();
	transform_reduce<<<blocks, THREADS>>>(arr, n, cnt, 100, ans_dev);
	timer.stop();
	std::cerr << "Reduction time: " << timer.milliseconds() << " ms\n";

	cudaMemcpy(&ans, ans_dev, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cnt);
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