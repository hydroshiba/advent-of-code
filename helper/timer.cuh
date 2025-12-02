#pragma once

class Timer {
private:
	cudaEvent_t begin;
	cudaEvent_t end;

public:
	Timer() {
		cudaEventCreate(&begin);
		cudaEventCreate(&end);
	}

	~Timer() {
		cudaEventDestroy(begin);
		cudaEventDestroy(end);
	}

	void start() {
		cudaEventRecord(begin, 0);
		cudaEventSynchronize(begin);
	}

	void stop(){
		cudaEventRecord(end, 0);
	}

	float milliseconds() {
		float elapsed;
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&elapsed, begin, end);
		return elapsed;
	}
};