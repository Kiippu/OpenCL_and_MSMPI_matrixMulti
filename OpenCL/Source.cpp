#include "mpi.h"
#include <ostream>
#include <iostream>
#include "Timer.h"
#include <vector>
//#include <omp.h>
#include <CL/cl.h>


#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <Windows.h>

/// N x N matrices
const int MAX_MATRIX_LENGTH = 16;
const int MAX_MATRIX_SIZE = MAX_MATRIX_LENGTH * MAX_MATRIX_LENGTH;
const int LOCAL_WORK_DIVISOR = 8;

#define 	MEM_SIZE                (128)
#define 	MAX_SOURCE_SIZE 	(0x100000)

/// rows and colums NxN
const int N = MAX_MATRIX_LENGTH;

MPI_Status status;

float matrix_0[N][N];
float matrix_1[N][N];
float matrix_final[N][N];


// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = 1.f;//rand() / (float)RAND_MAX;
}

void masterThread(int& i_iter, int& j_iter, int& k_iter, int& processorID, int& processorNum
	, int& totalProcessors, int& processorDestination, int& sourceID, int& matrixRows, int& rowOffset) {
	for (i_iter = 0; i_iter < N; i_iter++) {
		for (j_iter = 0; j_iter < N; j_iter++) {
			matrix_0[i_iter][j_iter] = 1.0f;
			matrix_1[i_iter][j_iter] = 1.0f;
			matrix_final[i_iter][j_iter] = 0.0f;
		}
	}

	/// split up rows for MPI processors
	matrixRows = N / totalProcessors;
	rowOffset = 0;

	/// set timer
	Timer::getInstance().addStartTime(eTimeLogType::TT_MULTIPLICATION_BEGIN, "Matric multiplication");

	/// send off array data to processors
	for (processorDestination = 1; processorDestination <= totalProcessors; processorDestination++)
	{
		MPI_Send(&rowOffset, 1, MPI_INT, processorDestination, 1, MPI_COMM_WORLD);
		MPI_Send(&matrixRows, 1, MPI_INT, processorDestination, 1, MPI_COMM_WORLD);
		MPI_Send(&matrix_0[rowOffset][0], matrixRows*N, MPI_FLOAT, processorDestination, 1, MPI_COMM_WORLD);
		MPI_Send(&matrix_1, N*N, MPI_FLOAT, processorDestination, 1, MPI_COMM_WORLD);
		/// set new rows to be sent to next iteration
		rowOffset = rowOffset + matrixRows;
	}

	/// catch all of the data sent be processed
	for (i_iter = 1; i_iter <= totalProcessors; i_iter++)
	{
		sourceID = i_iter;
		MPI_Recv(&rowOffset, 1, MPI_INT, sourceID, 2, MPI_COMM_WORLD, &status);
		MPI_Recv(&matrixRows, 1, MPI_INT, sourceID, 2, MPI_COMM_WORLD, &status);
		MPI_Recv(&matrix_final[rowOffset][0], matrixRows*N, MPI_FLOAT, sourceID, 2, MPI_COMM_WORLD, &status);
	}

	/// finish timer for multiplication
	Timer::getInstance().addFinishTime(eTimeLogType::TT_MULTIPLICATION_BEGIN);

	

};

void workerThread(int& i_iter, int& j_iter, int& k_iter, int& processorID, int& processorNum
	, int& totalProcessors, int& processorDestination, int& sourceID, int& matrixRows, int& rowOffset) {

	sourceID = 0;
	MPI_Recv(&rowOffset, 1, MPI_INT, sourceID, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&matrixRows, 1, MPI_INT, sourceID, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&matrix_0, matrixRows*N, MPI_FLOAT, sourceID, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&matrix_1, N*N, MPI_FLOAT, sourceID, 1, MPI_COMM_WORLD, &status);


	printf("\n\n thread :%d \n", processorID);


	/*printf("BEFORE - Matrix results:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			printf("%6.2f   ", matrix_final[i][j]);
		printf("\n");
	}*/




	//printf("matrixRows: %d\n", matrixRows);
	/// per process matrix multiplication 
	/*omp_set_dynamic(0);
	omp_set_num_threads(maxNumThreads);
	#pragma omp parallel for */
	/*for (k_iter = 0; k_iter < N; k_iter++)
	{
		for (i_iter = 0; i_iter < matrixRows; i_iter++)
		{
			for (j_iter = 0; j_iter < N; j_iter++)
				matrix_final[i_iter][k_iter] += matrix_0[i_iter][j_iter] * matrix_1[j_iter][k_iter];
		}
	}*/

	int matrixSize = MAX_MATRIX_SIZE / totalProcessors;

	////Allocate host memory for matrices A and B
	unsigned int matrix_0_mem_size = sizeof(float) * matrixSize;//(matrixRows*MAX_MATRIX_LENGTH);
	float* matrix_0_host_mem = (float*)malloc(matrix_0_mem_size);

	unsigned int matrix_1_mem_size = sizeof(float) * MAX_MATRIX_SIZE;
	float* matrix_1_host_mem = (float*)malloc(matrix_1_mem_size);


	for (k_iter = 0; k_iter < (matrixRows); k_iter++)
	{
		for (i_iter = 0; i_iter < MAX_MATRIX_LENGTH; i_iter++)
		{
			matrix_0_host_mem[(k_iter * MAX_MATRIX_LENGTH) + i_iter] = matrix_0[k_iter][i_iter];
			
		}
	}
	printf("\n\nmatrix_0_host_mem == matrixRows: %d, k:%d, i:%d\n", matrixRows, k_iter, i_iter);


	for (k_iter = 0; k_iter < MAX_MATRIX_LENGTH; k_iter++)
	{
		for (i_iter = 0; i_iter < MAX_MATRIX_LENGTH; i_iter++)
		{
			//matrix_0_host_mem[(k_iter * MAX_MATRIX_LENGTH) + i_iter] = matrix_0[k_iter][i_iter];
			matrix_1_host_mem[(k_iter * MAX_MATRIX_LENGTH) + i_iter] = matrix_1[k_iter][i_iter];
		}
	}
	printf("\n\nmatrix_1_host_mem == matrixRows: %d, k:%d, i:%d\n", matrixRows, k_iter, i_iter);

	///////////////////////////////////////////////////////////////////////////////////////

	int errMsg;                            // error code returned from api calls
	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;                   // compute kernel

	 // OpenCL device memory for matrices
	cl_mem matrix_0_mem;
	cl_mem matrix_1_mem;
	cl_mem matrix_2_mem;


	//Initialize host memory
	//randomMemInit(matrix_0_host_mem, MAX_MATRIX_SIZE);
	//randomMemInit(matrix_1_host_mem, MAX_MATRIX_SIZE);


	//Allocate host memory for the result C
	unsigned int matrix_2_mem_size = sizeof(float) * MAX_MATRIX_SIZE;
	float* matrix_2_host_mem = (float*)malloc(matrix_2_mem_size);

	printf("Initializing OpenCL device...\n");

	cl_uint dev_cnt = processorID;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	// Connect to a compute device
	int gpu = 1;
	errMsg = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (errMsg != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return;
	}

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &errMsg);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return;
	}

	// Create a command commands
	commands = clCreateCommandQueueWithProperties(context, device_id, 0, &errMsg);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return;
	}

	/* Load the source code containing the kernel */
	char string[128];
	FILE *fp;
	char fileName[] = "D://University//Uni 2019//Programming Paradigms//assignments//M3.T1P//OpenCL//OpenCL//vector_multiplication_kernel.cl";//"vector_add_kernel.cl";
	char *source_str;
	size_t source_size;

	fp = fopen(fileName, "r");
	if (!fp) {

		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Create Kernel Program from source
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &errMsg);
	if (errMsg != CL_SUCCESS) {
		printf("Failed to create OpenCL program from source %d\n", (int)errMsg);
		//goto error;
	}

	// Build the program executable
	errMsg = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (errMsg != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	//
	kernel = clCreateKernel(program, "mult_vec1_gpu", &errMsg);
	if (!kernel || errMsg != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Create the input and output arrays in device memory for our calculation
	matrix_2_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_2_mem_size, matrix_2_host_mem, &errMsg);
	matrix_0_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrix_0_mem_size, matrix_0_host_mem, &errMsg);
	matrix_1_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrix_1_mem_size, matrix_1_host_mem, &errMsg);

	if (!matrix_0_mem || !matrix_1_mem || !matrix_2_mem)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}

	printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n", MAX_MATRIX_LENGTH, MAX_MATRIX_LENGTH, MAX_MATRIX_LENGTH, MAX_MATRIX_LENGTH);

	//Launch OpenCL kernel
	size_t localWorkSize[2], globalWorkSize[2];

	errMsg = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matrix_2_mem);
	errMsg |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matrix_0_mem);
	errMsg |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&matrix_1_mem);
	errMsg |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&MAX_MATRIX_LENGTH);
	errMsg |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&rowOffset);

	if (errMsg != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", errMsg);
		exit(1);
	}

	if ((MAX_MATRIX_LENGTH % LOCAL_WORK_DIVISOR) != 0)
	{
		float  result = ((float)MAX_MATRIX_LENGTH / (float)LOCAL_WORK_DIVISOR);
		printf("Error: MAX_MATRIX_LENGTH is not divible by LOCAL_WORK_DIVISOR!  %d / %d == %f\n", MAX_MATRIX_LENGTH, LOCAL_WORK_DIVISOR, result);
		exit(1);
	}

	localWorkSize[0] = MAX_MATRIX_LENGTH / LOCAL_WORK_DIVISOR;
	localWorkSize[1] = MAX_MATRIX_LENGTH;
	globalWorkSize[0] = MAX_MATRIX_LENGTH;
	globalWorkSize[1] = MAX_MATRIX_LENGTH;

	errMsg = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	if (errMsg != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", errMsg);
		exit(1);
	}

	//Retrieve result from device
	errMsg = clEnqueueReadBuffer(commands, matrix_2_mem, CL_TRUE, 0, matrix_2_mem_size, matrix_2_host_mem, 0, NULL, NULL);
	printf("just finshed opencl ID %d\n", processorID);
	if (errMsg != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", errMsg);
		exit(1);
	}

	///////////////////////////////////////////////////////////////////////

	printf("rowOffset: %d\n", rowOffset);
	/*for (int k = rowOffset; k < MAX_MATRIX_LENGTH; k++)
	{
		for (int i = 0; i < matrixRows; i++)
		{
			matrix_final[k][i] = matrix_2_host_mem[(k*MAX_MATRIX_LENGTH) + i];
		}
	}*/
	/// this does first 4 columns
	/*for (int k = 0; k < N; k++)
	{
		for (int i = 0; i < N; i++)
		{
			matrix_final[k][i] = matrix_2_host_mem[(k*MAX_MATRIX_LENGTH) + i];
		}
	}*/
	//if (processorID == 2) {
	//printf("matrixRows: %d, k:%d, i:%d\n", matrixRows, k_iter, i_iter);
		for (k_iter = 0; k_iter < (N); k_iter++)
		{
			for (i_iter = 0; i_iter < N; i_iter++)
			{
				matrix_final[i_iter][k_iter] = matrix_2_host_mem[(k_iter*MAX_MATRIX_LENGTH) + i_iter];
				//printf("i:%d, k:%d, ind:%d\t", i_iter, k_iter, (k_iter*MAX_MATRIX_LENGTH) + i_iter);
			}
		}
		//printf("matrixRows: %d, k:%d, i:%d\n", matrixRows, k_iter, i_iter);
	//}

	///////////////////////////////////////////////////////////////////////
	/*int poo2 = (MAX_MATRIX_LENGTH / totalProcessors);
	int poo = (rowOffset*MAX_MATRIX_LENGTH) + (MAX_MATRIX_SIZE / totalProcessors);
	printf("\n\npoo: %d\n\n", poo);

	for (int k = (rowOffset); k < (rowOffset+ poo2); k++)
	{
		for (int i = 0; i < N; i++)
		{
			matrix_final[k][i] = matrix_2_host_mem[(k*MAX_MATRIX_LENGTH) + i];
			printf("i: %d = %d, ", (k*MAX_MATRIX_LENGTH) + i), matrix_2_host_mem[(k*MAX_MATRIX_LENGTH) + i];
		}
	}
*/
	//////////////////////////////////////////////////////////////////////

	//print out the results
	/*printf("\n\nMatrix openCL (Results)\n");
	int i;
	for (i = 0; i < MAX_MATRIX_SIZE; i++)
	{
		printf("%f ", matrix_2_host_mem[i]);
		if (((i + 1) % MAX_MATRIX_LENGTH) == 0)
			printf("\n");
	}
	printf("\n");*/

	/*printf("\n\nMatrix mpi:\n");
	for (i_iter = 0; i_iter < N; i_iter++) {
		for (j_iter = 0; j_iter < N; j_iter++)
			printf("%6.2f   ", matrix_final[i_iter][j_iter]);
		printf("\n");
	}*/


	printf("Matrix multiplication completed...\n");

	//Shutdown and cleanup
	free(matrix_0_host_mem);
	free(matrix_1_host_mem);
	free(matrix_2_host_mem);

	clReleaseMemObject(matrix_0_mem);
	clReleaseMemObject(matrix_2_mem);
	clReleaseMemObject(matrix_1_mem);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);


	///////////////////////////////////////////////////////////////////////////////////////


	/// sending matrix data back to the master thread
	MPI_Send(&rowOffset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
	MPI_Send(&matrixRows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
	MPI_Send(&matrix_final, matrixRows*N, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);

};


int main(int argc, char **argv)
{
	int processorNum;
	int processorID;
	int totalProcessors;
	int processorDestination;
	int sourceID;
	int matrixRows;
	int rowOffset;
	int i_iter, j_iter, k_iter;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &processorID);
	MPI_Comm_size(MPI_COMM_WORLD, &processorNum);

	totalProcessors = processorNum - 1;
	printf("totalProcessors: %d", totalProcessors);

	/// Master Process 
	// in charge of sending and setting array data to processors
	if (processorID == 0) {
		masterThread(i_iter, j_iter, k_iter, processorID, processorNum, totalProcessors, processorDestination, sourceID, matrixRows, rowOffset);

		/// print all results for the matrix
		// uncommment to see results
		printf("Matrix results:\n");
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++)
				printf("%6.2f, ", matrix_final[i][j]);
			printf("\n");
		}

		/// print time taken
		Timer::getInstance().printFinalTimeSheet();

	}

	/// All processors but master thread
	if (processorID > 0) {
		workerThread(i_iter, j_iter, k_iter, processorID, processorNum, totalProcessors, processorDestination, sourceID, matrixRows, rowOffset);
	}
	
	MPI_Finalize();

	return 0;
}

