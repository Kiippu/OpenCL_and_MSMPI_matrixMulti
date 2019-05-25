//////////////////////////////////////////////////////////////////////////////////
//
//#include <fcntl.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include <math.h>
////#include <unistd.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <CL/cl.h>
//#include <stdbool.h>
//
//////////////////////////////////////////////////////////////////////////////////
//#define WA 16
//#define HA 16
//#define WB 16
//
//#define HB WA
//#define WC WB
//#define HC HA
//
//
///// N x N matrices
//const int MAX_MATRIX_LENGTH = 8;
//const int MAX_MATRIX_SIZE = MAX_MATRIX_LENGTH * MAX_MATRIX_LENGTH;
//const int LOCAL_WORK_DIVISOR = 4;
//
//#define 	MEM_SIZE                (128)
//#define 	MAX_SOURCE_SIZE 	(0x100000)
//////////////////////////////////////////////////////////////////////////////////
//
// 
//// Allocates a matrix with random float entries.
//void randomMemInit(float* data, int size)
//{
//   for (int i = 0; i < size; ++i)
//	   data[i] = 1.f;//rand() / (float)RAND_MAX;
//}
//
//int main(int argc, char** argv)
//{
//
//   int errMsg;                            // error code returned from api calls
//   cl_device_id device_id;             // compute device id 
//   cl_context context;                 // compute context
//   cl_command_queue commands;          // compute command queue
//   cl_program program;                 // compute program
//   cl_kernel kernel;                   // compute kernel
//
//    // OpenCL device memory for matrices
//   cl_mem matrix_0_mem;
//   cl_mem matrix_1_mem;
//   cl_mem matrix_2_mem;
//
//   //Allocate host memory for matrices A and B
//   unsigned int matrix_0_mem_size = sizeof(float) * MAX_MATRIX_SIZE;
//   float* matrix_0_host_mem = (float*) malloc(matrix_0_mem_size);
// 
//   unsigned int matrix_1_mem_size = sizeof(float) * MAX_MATRIX_SIZE;
//   float* matrix_1_host_mem = (float*) malloc(matrix_1_mem_size);
//
//   //Initialize host memory
//   randomMemInit(matrix_0_host_mem, MAX_MATRIX_SIZE);
//   randomMemInit(matrix_1_host_mem, MAX_MATRIX_SIZE);
// 
//   //Allocate host memory for the result C
//   unsigned int matrix_2_mem_size = sizeof(float) * MAX_MATRIX_SIZE;
//   float* matrix_2_host_mem = (float*) malloc(matrix_2_mem_size);
//  
//   printf("Initializing OpenCL device...\n"); 
//
//   cl_uint dev_cnt = 0;
//   clGetPlatformIDs(0, 0, &dev_cnt);
//	
//   cl_platform_id platform_ids[100];
//   clGetPlatformIDs(dev_cnt, platform_ids, NULL);
//	
//   // Connect to a compute device
//   int gpu = 1;
//   errMsg = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
//   if (errMsg != CL_SUCCESS)
//   {
//       printf("Error: Failed to create a device group!\n");
//       return EXIT_FAILURE;
//   }
//  
//   // Create a compute context 
//   context = clCreateContext(0, 1, &device_id, NULL, NULL, &errMsg);
//   if (!context)
//   {
//       printf("Error: Failed to create a compute context!\n");
//       return EXIT_FAILURE;
//   }
//
//   // Create a command commands
//   commands = clCreateCommandQueueWithProperties(context, device_id, 0, &errMsg);
//   if (!commands)
//   {
//       printf("Error: Failed to create a command commands!\n");
//       return EXIT_FAILURE;
//   }
//
//   /* Load the source code containing the kernel */
//   char string[128];
//   FILE *fp;
//   char fileName[] = "vector_multiplication_kernel.cl";//"vector_add_kernel.cl";
//   char *source_str;
//   size_t source_size;
//
//   fp = fopen(fileName, "r");
//   if (!fp) {
//
//	   fprintf(stderr, "Failed to load kernel.\n");
//	   exit(1);
//   }
//   source_str = (char*)malloc(MAX_SOURCE_SIZE);
//   source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
//   fclose(fp);
//
//   	// Create Kernel Program from source
//   	program = clCreateProgramWithSource(context, 1, (const char **)&source_str,(const size_t *)&source_size, &errMsg);
//   	if (errMsg != CL_SUCCESS) {
//   		printf("Failed to create OpenCL program from source %d\n", (int)errMsg);
//   		//goto error;
//   	}
//
//   // Build the program executable
//   errMsg = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
//   if (errMsg != CL_SUCCESS)
//   {
//       size_t len;
//       char buffer[2048];
//       printf("Error: Failed to build program executable!\n");
//       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
//       printf("%s\n", buffer);
//       exit(1);
//   }
//
//   // Create the compute kernel in the program we wish to run
//   //
//   kernel = clCreateKernel(program, "mult_vec_gpu", &errMsg);
//   if (!kernel || errMsg != CL_SUCCESS)
//   {
//       printf("Error: Failed to create compute kernel!\n");
//       exit(1);
//   }
//
//   // Create the input and output arrays in device memory for our calculation
//   matrix_2_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_0_mem_size, NULL, &errMsg);
//   matrix_0_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrix_0_mem_size, matrix_0_host_mem, &errMsg);
//   matrix_1_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrix_1_mem_size, matrix_1_host_mem, &errMsg);
//
//   if (!matrix_0_mem || !matrix_1_mem || !matrix_2_mem)
//   {
//       printf("Error: Failed to allocate device memory!\n");
//       exit(1);
//   }    
//    
//   printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n", WA,HA,WB,HB); 
//
//   //Launch OpenCL kernel
//   size_t localWorkSize[2], globalWorkSize[2];
// 
//   errMsg = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matrix_2_mem);
//   errMsg |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matrix_0_mem);
//   errMsg |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&matrix_1_mem);
//   errMsg |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&MAX_MATRIX_LENGTH);
//   errMsg |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&MAX_MATRIX_LENGTH);
//
//   if (errMsg != CL_SUCCESS)
//   {
//       printf("Error: Failed to set kernel arguments! %d\n", errMsg);
//       exit(1);
//   }
// 
//   if ((MAX_MATRIX_LENGTH % LOCAL_WORK_DIVISOR) != 0) 
//   {
//	   float  result = ((float)MAX_MATRIX_LENGTH / (float)LOCAL_WORK_DIVISOR);
//	   printf("Error: MAX_MATRIX_LENGTH is not divible by LOCAL_WORK_DIVISOR!  %d / %d == %f\n", MAX_MATRIX_LENGTH, LOCAL_WORK_DIVISOR, result);
//	   exit(1);
//   }
//
//   localWorkSize[0] = MAX_MATRIX_LENGTH / LOCAL_WORK_DIVISOR;
//   localWorkSize[1] = MAX_MATRIX_LENGTH;
//   globalWorkSize[0] = MAX_MATRIX_LENGTH;
//   globalWorkSize[1] = MAX_MATRIX_LENGTH;
// 
//   errMsg = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//
//   if (errMsg != CL_SUCCESS)
//   {
//       printf("Error: Failed to execute kernel! %d\n", errMsg);
//       exit(1);
//   }
// 
//   //Retrieve result from device
//   errMsg = clEnqueueReadBuffer(commands, matrix_2_mem, CL_TRUE, 0, matrix_2_mem_size, matrix_2_host_mem, 0, NULL, NULL);
//
//   if (errMsg != CL_SUCCESS)
//   {
//       printf("Error: Failed to read output array! %d\n", errMsg);
//       exit(1);
//   }
// 
//   //print out the results
//
//   printf("\n\nMatrix C (Results)\n");
//   int i;
//   for(i = 0; i < MAX_MATRIX_SIZE; i++)
//   {
//      printf("%f ", matrix_2_host_mem[i]);
//      if(((i + 1) % MAX_MATRIX_LENGTH) == 0)
//      printf("\n");
//   }
//   printf("\n");
//
//  
//   printf("Matrix multiplication completed...\n"); 
//
//   //Shutdown and cleanup
//   free(matrix_0_host_mem);
//   free(matrix_1_host_mem);
//   free(matrix_2_host_mem);
// 
//   clReleaseMemObject(matrix_0_mem);
//   clReleaseMemObject(matrix_2_mem);
//   clReleaseMemObject(matrix_1_mem);
//
//   clReleaseProgram(program);
//   clReleaseKernel(kernel);
//   clReleaseCommandQueue(commands);
//   clReleaseContext(context);
//
//   return 0;
//}
//
//
////
////#include <stdio.h>
////#include <stdlib.h>
////#include <math.h>
////
////#include <time.h>
////#include <CL/cl.h>
////
////#define         PRINT_LINE(title)       printf("\n========== %s ==========\n", title);
////
////
////void init_vec(int *vec, int len, int set_one_flag) {
////	for (int i = 0; i < len; i++) {
////		if (set_one_flag)
////			vec[i] = 1;
////		else
////			vec[i] = 0;
////	}
////}
////
////void rand_vec(int *vec, int len) {
////	srand((unsigned)time(0));
////	for (int i = 0; i < len; i++) {
////		vec[i] = 1;//rand() % 3;
////	}
////}
////
////void add_vec_cpu(const int *a, const int *b, int *res, const int len) {
////	for (int i = 0; i < len; i++) {
////		res[i] = a[i] * b[i];
////	}
////}
////
////void print_vec(int *vec, int len) {
////	for (int i = 0; i < len; i++) {
////		printf("%d ", vec[i]);
////	}
////	printf("\n");
////}
////
////void check_result(int *v1, int *v2, int len) {
////	int correct_num = 0;
////	for (int i = 0; i < len; i++) {
////		if (v1[i] == v2[i]) {
////			correct_num += 1;
////		}
////	}
////	printf("correct rate: %d / %d , %1.2f\n", correct_num, len, (float)correct_num / len);
////}
////
////int main(void) {
////
////	double duration;
////	srand((unsigned)time(NULL));
////
////	/* generate vector a and b */
////	/*int len = 64;
////	int *a, *b, *c, *c_d;
////	a = (int *)malloc(len * sizeof(int));
////	b = (int *)malloc(len * sizeof(int));
////	c = (int *)malloc(len * sizeof(int));
////	c_d = (int *)malloc(len * sizeof(int));*/
////	//size_t data_size = len * sizeof(int);
////
////
////	/*generate matrix a and b */
////	int* matrix_0;
////	int* matrix_1;
////	int* matrix_2;
////	int* matrix_21;
////
////	//for (size_t i = 0; i < MAX_MATRIX_LENGTH; i++)
////	//{
////	matrix_0 = (int *)malloc(MAX_MATRIX_SIZE * sizeof(int));
////	matrix_1 = (int *)malloc(MAX_MATRIX_SIZE * sizeof(int));
////	matrix_2 = (int *)malloc(MAX_MATRIX_SIZE * sizeof(int));
////	matrix_21 = (int *)malloc(MAX_MATRIX_SIZE * sizeof(int));
////	init_vec(matrix_0, MAX_MATRIX_SIZE, 1);
////	rand_vec(matrix_1, MAX_MATRIX_SIZE);
////	init_vec(matrix_2, MAX_MATRIX_SIZE, 0);
////	//}
////	size_t data_size = MAX_MATRIX_SIZE * sizeof(int);
////	add_vec_cpu(matrix_0, matrix_1, matrix_2, MAX_MATRIX_SIZE);
////
////	//PRINT_LINE("INIT VALUE");
////	/* vector addition, cpu version */
////	printf("a: ");
////	//init_vec(matrix_0, MAX_MATRIX_SIZE, 1);
////	print_vec(matrix_0, MAX_MATRIX_SIZE);
////
////	printf("b: ");
////	//rand_vec(matrix_1, MAX_MATRIX_SIZE);
////	print_vec(matrix_1, MAX_MATRIX_SIZE);
////
////	printf("c: ");
////	//init_vec(matrix_2, MAX_MATRIX_SIZE, 0);
////	add_vec_cpu(matrix_0, matrix_1, matrix_2, MAX_MATRIX_SIZE);
////	print_vec(matrix_2, MAX_MATRIX_SIZE);
////
////	/* vector addition, gpu version  */
////	cl_mem a_buff, b_buff, c_buff;
////	a_buff = b_buff = c_buff = NULL;
////
////	cl_platform_id platform_id = NULL;
////	cl_uint ret_num_platforms;
////
////	cl_device_id device_id = NULL;
////	cl_uint ret_num_devices;
////
////	cl_context context = NULL;
////	cl_kernel kernel = NULL;
////	cl_program program = NULL;
////
////	cl_command_queue command_queue = NULL;
////	cl_int ret;
////
////	/* Load the source code containing the kernel */
////	char string[MEM_SIZE];
////	FILE *fp;
////	char fileName[] = "vector_multiplication_kernel.cl";//"vector_add_kernel.cl";
////	char *source_str;
////	size_t source_size;
////
////	fp = fopen(fileName, "r");
////	if (!fp) {
////
////		fprintf(stderr, "Failed to load kernel.\n");
////		exit(1);
////	}
////	source_str = (char*)malloc(MAX_SOURCE_SIZE);
////	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
////	fclose(fp);
////
////	// Platform
////	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to get platform ID.\n");
////		goto error;
////	}
////	// Device
////	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to get device ID.\n");
////		goto error;
////	}
////	// Context
////	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);//&ret);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to create OpenCL context.\n");
////		goto error;
////	}
////	command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to create command queue %d\n", (int)ret);
////		goto error;
////	}
////	// Memory Buffer
////	//a_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
////	//b_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
////	//c_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &ret);
////	// Create the input and output arrays in device memory for our calculation
////	c_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, NULL, &ret);
////	a_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data_size, matrix_0, &ret);
////	b_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data_size, matrix_1, &ret);
////
////	ret = clEnqueueWriteBuffer(command_queue, a_buff, CL_TRUE, 0, data_size, (void *)matrix_0, 0, NULL, NULL);
////	ret |= clEnqueueWriteBuffer(command_queue, b_buff, CL_TRUE, 0, data_size, (void *)matrix_1, 0, NULL, NULL);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to copy date from host to device: %d\n", (int)ret);
////		goto error;
////	}
////	// Create Kernel Program from source
////	program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
////		(const size_t *)&source_size, &ret);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to create OpenCL program from source %d\n", (int)ret);
////		goto error;
////	}
////	// Build Kernel Program
////	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to build program %d\n", (int)ret);
////		char build_log[16348];
////		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
////		printf("Error in kernel: %s\n", build_log);
////		goto error;
////	}
////	// Create OpenCL Kernel
////	kernel = clCreateKernel(program, "mult_vec_gpu", &ret);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to create kernel %d\n", (int)ret);
////		goto error;
////	}
////	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_buff);
////	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_buff);
////	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_buff);
////	ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&MAX_MATRIX_LENGTH);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to set kernel arguments %d\n", (int)ret);
////		goto error;
////	}
////
////	/* Execute OpenCL Kernel */
////	// executed using a single work-item
////	// ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);
////
////	size_t global_work_size[2], local_work_size[2];
////	// Number of work items in each local work group  
////	local_work_size[0] = MAX_MATRIX_LENGTH;
////	local_work_size[1] = MAX_MATRIX_LENGTH;
////	// Number of total work items - localSize must be devisor  
////	global_work_size[0] = MAX_MATRIX_SIZE;
////	global_work_size[1] = MAX_MATRIX_SIZE;//= (size_t)ceil(MAX_MATRIX_SIZE / (int)local_work_size) * local_work_size;
////
////	//size_t local_work_size[2] = { 8, 8 };
////	//size_t global_work_size[2] = { 1, len };
////	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to execute kernel for execution %d\n", (int)ret);
////		goto error;
////	}
////
////	init_vec(matrix_21, MAX_MATRIX_SIZE, 0);
////	/* Copy results from the memory buffer */
////	ret = clEnqueueReadBuffer(command_queue, c_buff, CL_TRUE, 0, data_size, (void *)matrix_21, 0, NULL, NULL);
////	if (ret != CL_SUCCESS) {
////		printf("Failed to copy data from device to host %d\n", (int)ret);
////		goto error;
////	}
////
////	/* Display Result */
////	PRINT_LINE("CHECK RESULT cpu-verison && gpu-version");
////	printf("matrix_21[0]: ");
////	print_vec(matrix_21, MAX_MATRIX_SIZE);
////	check_result(matrix_2, matrix_21, MAX_MATRIX_SIZE);
////	printf("MAX_MATRIX_LENGTH-1=%d, matrix_21[0][%d]==c[%d]: %d, matrix_21[0][%d]=%d, c[%d]=%d \n", MAX_MATRIX_SIZE - 1, MAX_MATRIX_SIZE - 1, MAX_MATRIX_SIZE - 1, matrix_21[MAX_MATRIX_SIZE - 1] == matrix_2[MAX_MATRIX_SIZE - 1], MAX_MATRIX_SIZE - 1, matrix_21[MAX_MATRIX_SIZE - 1], MAX_MATRIX_SIZE - 1, matrix_2[MAX_MATRIX_SIZE - 1]);
////
////	PRINT_LINE("CHECK RESULT ELEMENT BY ELEMENT");
////	printf("idx  matrix_2[0]  matrix_21[0]\n");
////	for (int i = 0; i < MAX_MATRIX_SIZE; i++) {
////		printf("%2d %2d %2d \n", i, matrix_2[i], matrix_21[i]);
////	}
////
////	/* Finalization */
////error:
////
////	/* free device resources */
////	clFlush(command_queue);
////	clFinish(command_queue);
////	clReleaseKernel(kernel);
////	clReleaseProgram(program);
////
////	clReleaseMemObject(a_buff);
////	clReleaseMemObject(b_buff);
////	clReleaseMemObject(c_buff);
////
////	clReleaseCommandQueue(command_queue);
////	clReleaseContext(context);
////
////	/* free host resources */
////	free(source_str);
////	free(matrix_0);
////	free(matrix_1);
////	free(matrix_2);
////
////	return 0;
////}