#include <cstdio>
#include <random>
#include <map>
#include <vector>
#include <string>
#include <mpi.h>

const long SIZE = 64;

struct Result
{
	long key;
	float r;
};

long random(const int &min, const int &max)
{
	static std::mt19937 generator(117);
	std::uniform_int_distribution<long> distribution(min, max);
	return distribution(generator);
};

void init(auto &M, const long c1, const long c2, const long key)
{
	for (long i = 0; i < c1; ++i)
		for (long j = 0; j < c2; ++j)
			M[i][j] = (key - i - j) / static_cast<float>(SIZE);
}

// matrix multiplication:  C = A x B  A[c1][c2] B[c2][c1] C[c1][c1]
// mm returns the sum of the elements of the C matrix
auto mm(const auto &A, const auto &B, const long c1, const long c2)
{
	float sum{0};
	for (long i = 0; i < c1; i++)
	{
		for (long j = 0; j < c1; j++)
		{
			auto accum = float(0.0);
			for (long k = 0; k < c2; k++)
			{
				accum += A[i][k] * B[k][j];
			}
			sum += accum;
		}
	}
	return sum;
}

// initialize two matrices with the computed values of the keys
// and execute a matrix multiplication between the two matrices
// to obtain the sum of the elements of the result matrix
float compute(const long c1, const long c2, long key1, long key2)
{

	std::vector<std::vector<float>> A(c1, std::vector<float>(c2, 0.0));
	std::vector<std::vector<float>> B(c2, std::vector<float>(c1, 0.0));

	init(A, c1, c2, key1);
	init(B, c2, c1, key2);
	auto r = mm(A, B, c1, c2);
	return r;
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::printf("use: %s nkeys length [print(0|1)]\n", argv[0]);
		std::printf("     print: 0 disabled, 1 enabled\n");
		return -1;
	}

	long nkeys = std::stol(argv[1]); // total number of keys
	// length is the "stream length", i.e. the number of random key pairs generated
	long length = std::stol(argv[2]);
	bool print = false;
	if (argc == 4)
		print = (std::stoi(argv[3]) == 1) ? true : false;

	long key1, key2;

	std::map<long, long> map;
	for (long i = 0; i < nkeys; ++i)
		map[i] = 0;

	// define a map to keep pending request
	std::map<long, bool> pending;
	for (long i = 0; i < nkeys; ++i)
		pending[i] = false;

	std::vector<float> V(nkeys, 0);

	int myrank;
	int numP;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numP);

	if (numP < 2)
	{
		std::printf("At least 2 processes are required\n");
		MPI_Finalize();
		return -1;
	}

	// define new data type to send one long and one float
	MPI_Datatype RESULT_T;
	Result result;
	int blocklen[] = {1, 1};
	MPI_Datatype oldtypes[] = {MPI_LONG, MPI_FLOAT};
	MPI_Aint first_addr, secon_addr, displs[2];
	MPI_Get_address(&result.key, &first_addr);
	MPI_Get_address(&result.r, &secon_addr);
	displs[0] = 0;
	displs[1] = secon_addr - first_addr;
	MPI_Type_create_struct(2, blocklen, displs, oldtypes, &RESULT_T);
	MPI_Type_commit(&RESULT_T);

	// just to make sure that all processes start at the same time
	MPI_Barrier(MPI_COMM_WORLD);

	if (myrank == 0)
	{
		// variable to distribute the keys between the processes
		int round = 0;

		// start the timer
		double start = MPI_Wtime();

		for (int i = 0; i < length; ++i)
		{
			key1 = random(0, nkeys - 1); // value in [0,nkeys[
			key2 = random(0, nkeys - 1); // value in [0,nkeys[

			if (key1 == key2) // only distinct values in the pair
				key1 = (key1 + 1) % nkeys;

			// check if key1 has a pending request
			while (pending[key1] || pending[key2])
			{
				// collect available results
				Result result;
				MPI_Recv(&result, 1, RESULT_T, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				float r1 = result.r;
				V[result.key] += r1;
				pending[result.key] = false;
				// reset
				auto _r1 = static_cast<unsigned long>(r1) % SIZE;
				map[result.key] = (_r1 > (SIZE / 2)) ? 0 : _r1;
			}

			map[key1]++; // count the number of key1 keys
			map[key2]++; // count the number of key2 keys

			// if key1 reaches the SIZE limit, send a request to a process to compute the result
			if (map[key1] == SIZE && map[key2] != 0)
			{
				// send the request to the computing process
				int computing_P = ((++round) % (numP - 1)) + 1;
				long data[4] = {map[key1], map[key2], key1, key2};
				MPI_Send(&data, 4, MPI_LONG, computing_P, 1, MPI_COMM_WORLD);
				pending[key1] = true;
			}
			// if key2 reaches the SIZE limit, send a request to a process to compute the result
			if (map[key2] == SIZE && map[key1] != 0)
			{
				// send the request to the computing process
				int computing_P = ((++round) % (numP - 1)) + 1;
				long data[4] = {map[key2], map[key1], key2, key1};
				MPI_Send(&data, 4, MPI_LONG, computing_P, 1, MPI_COMM_WORLD);
				pending[key2] = true;
			}
		}

		// reset the pending requests
		for (long i = 0; i < nkeys; ++i)
		{
			while (pending[i])
			{
				Result result;
				MPI_Recv(&result, 1, RESULT_T, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				float r1 = result.r;
				V[result.key] += r1;
				pending[result.key] = false;
				// reset
				auto _r1 = static_cast<unsigned long>(r1) % SIZE;
				map[result.key] = (_r1 > (SIZE / 2)) ? 0 : _r1;
			}
		}

		round = 0;
		// compute the last values
		for (long i = 0; i < nkeys; ++i)
		{
			for (long j = 0; j < nkeys; ++j)
			{
				if (i == j) continue;
				if (map[i] > 0 && map[j] > 0)
				{
					// send the request to the computing process
					int computing_P = ((++round) % (numP - 1)) + 1;
					long data1[4] = {map[i], map[j], i, j};
					MPI_Send(data1, 4, MPI_LONG, computing_P, 1, MPI_COMM_WORLD);

					// send the request to the computing process
					computing_P = ((++round) % (numP - 1)) + 1;
					long data2[4] = {map[j], map[i], j, i};
					MPI_Send(data2, 4, MPI_LONG, computing_P, 1, MPI_COMM_WORLD);
				}
			}
		}

		round = 0;
		// wait for the pending requests
		for (long i = 0; i < nkeys; ++i)
		{
			for (long j = 0; j < nkeys; ++j)
			{
				if (i == j) continue;
				if (map[i] > 0 && map[j] > 0)
				{
					int computing_P = ((++round) % (numP - 1)) + 1;
					Result result;
					MPI_Recv(&result, 1, RESULT_T, computing_P, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					V[result.key] += result.r;

					computing_P = ((++round) % (numP - 1)) + 1;
					MPI_Recv(&result, 1, RESULT_T, computing_P, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					V[result.key] += result.r;
				}
			}
		}

		// stop the timer
		double end = MPI_Wtime();

		// terminate the other processes
		for (int p = 1; p < numP; p++)
			MPI_Send(NULL, 0, MPI_INT, p, 9, MPI_COMM_WORLD);

		// print the elapsed time
		std::printf("Elapsed time: %f, with %d proc, keys= %ld, length=%ld\n", end - start, numP, nkeys, length);

		// printing the results
		if (print)
		{
			for (long i = 0; i < nkeys; ++i)
				std::printf("key %ld : %f\n", i, V[i]);
		}
	}
	else
	{
		// computing process
		MPI_Status status;
		long data[4];

		while (true)
		{
			MPI_Recv(&data, 4, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG == 9)
			{
				break;
			}
			long c1 = data[0];
			long c2 = data[1];
			long key1 = data[2];
			long key2 = data[3];

			float r = compute(c1, c2, key1, key2);

			// send the result
			Result result;
			result.key = key1;
			result.r = r;
			MPI_Send(&result, 1, RESULT_T, 0, 2, MPI_COMM_WORLD);

			/*
			MPI_Request request;
			Result *result = new Result;
			result->key = key1;
			result->r = r;
			MPI_Isend(result, 1, RESULT_T, 0, 2, MPI_COMM_WORLD, &request);
			MPI_Request_free(&request);
			*/
		}
	}

	MPI_Type_free(&RESULT_T);
	MPI_Finalize();
	return 0;
}
