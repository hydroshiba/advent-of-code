#include <iostream>
#include "timer.cuh"

__global__ void chunk_rotate(int* histogram) {

}

int main() {
	std::vector<int> vec;
	char c;

	while(std::cin >> c) {
		int num;
		std::cin >> num;
		vec.push_back(c == 'L' ? -num : num);
	}

	int start = 50;

}