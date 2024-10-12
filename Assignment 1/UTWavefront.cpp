//
// Parallel code of the first SPM Assignment a.a. 23/24.
//
// compile:
// g++ -std=c++20 -O3 -march=native -I include/ UTWavefront.cpp -o UTW
//
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <cassert>
#include <barrier>
#include <hpc_helpers.hpp>
#include <threadPool.hpp>

int random(const int &min, const int &max)
{
	static std::mt19937 generator(117);
	std::uniform_int_distribution<int> distribution(min, max);
	return distribution(generator);
};

// emulate some work
void work(std::chrono::microseconds w)
{
	auto end = std::chrono::steady_clock::now() + w;
	while (std::chrono::steady_clock::now() < end);
}

void wavefront(const std::vector<int> &M, const uint64_t &N, const uint64_t &n_threads)
{
	// initialize matrix indexes
	int diag_k = 0;
	std::atomic<int> elem_index(0);

	auto on_completion = [&]() -> void  // update indexes when one diagonal is done (with barrier)
	{
		diag_k++;
		elem_index.store(0);
	};

	// create a barrier
	std::barrier barrier(n_threads, on_completion);

	auto wavefront_inner = [&]() -> void
	{
		int i = 0;
		while (diag_k < N)	// for each diagonal
		{
			while ((i = elem_index.fetch_add(1, std::memory_order_seq_cst)) < (N - diag_k))  // while there are elem. in the diag. not computed
			{
				work(std::chrono::microseconds(M[i * N + (i + diag_k)]));
			}

			// no more diagonal to compute, thread can exit
			if (diag_k == N-1)
				return;

			// wait all the diagonal elements to be computed
			barrier.arrive_and_wait();
		}
	};

	// create threads
	std::vector<std::thread> threads;
	for (uint64_t id = 0; id < n_threads; id++)
		threads.emplace_back(wavefront_inner);

	// wait for the threads to finish
	for (auto &thread : threads)
		thread.join();
};

int main(int argc, char *argv[])
{
	uint64_t N = 512;		// default size of the matrix (NxN)
	uint64_t n_threads = 1; // default number of threads
	int min = 0;			// default minimum time (in microseconds)
	int max = 1000;			// default maximum time (in microseconds)

	if (argc != 1 && argc != 2 && argc != 3 && argc != 5)
	{
		std::printf("Use: %s [n_threads N min max]\n", argv[0]);
		std::printf("     n_threads number of threads\n");
		std::printf("     N size of the square matrix\n");
		std::printf("     min waiting time (us)\n");
		std::printf("     max waiting time (us)\n");

		return -1;
	}
	if (argc > 1)
	{
		n_threads = std::stol(argv[1]);
		
		if (argc > 2)
		{
			N = std::stol(argv[2]);
		}
		if (argc > 4)
		{
			min = std::stol(argv[3]);
			max = std::stol(argv[4]);
		}
	}

	// allocate the matrix
	std::vector<int> M(N * N, -1);

	uint64_t expected_totaltime = 0;

	// init function
	auto init = [&]()
	{
		for (uint64_t k = 0; k < N; ++k)
		{
			for (uint64_t i = 0; i < (N - k); ++i)
			{
				int t = random(min, max);
				M[i * N + (i + k)] = t;
				expected_totaltime += t;
			}
		}
	};

	init();

	std::printf("\nConfiguration: %lu threads, N = %lu, min = %d, max = %d\n", n_threads, N, min, max);
	std::printf("Estimated sequential compute time ~ %f (ms)\n", expected_totaltime / 1000.0);
	std::printf("Estimated optimal parallel compute time ~ %f (ms)\n", expected_totaltime / (1000.0 * n_threads));

	TIMERSTART(wavefront);
	wavefront(M, N, n_threads);
	TIMERSTOP(wavefront);

	return 0;
}
