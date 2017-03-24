#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// includes, kernels
#include "2Dconvolution_kernel.cu"

#define PRINT_RESULTS 0
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
float ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
float OptimizedConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
int checkResults(float *, float *, int, float);

__constant__ float cM[KERNEL_SIZE*KERNEL_SIZE];

__global__ void OptimizedConvolutionKernel(Matrix N, Matrix P)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0;

    unsigned int mBeg = i < 2 ? 2-i: 0;
    unsigned int mEnd = i>(MATRIX_SIZE-3) ? MATRIX_SIZE-i+2  : 5;
    unsigned int nBeg = j < 2 ? 2-j : 0;
    unsigned int nEnd = j > (MATRIX_SIZE-3) ? MATRIX_SIZE-j+2  : 5;

    for(int m=mBeg; m < mEnd; m++)
        for(int n=nBeg; n < nEnd; n++)
            sum += cM[KERNEL_SIZE*m + n] * N.elements[MATRIX_SIZE*(i + m-2) + ((j + n-2))];

    P.elements[ MATRIX_SIZE*i + j] = (float)sum;

}

//__constant__ float cN[MATRIX_SIZE*MATRIX_SIZE];
int main(int argc, char** argv) 
{

	Matrix  A;
	Matrix  B;
	Matrix  C;
    Matrix C_opt;	

	srand(time(NULL));
    struct timeval start1,stop1,start2,stop2,start3,stop3;

	// Allocate and initialize the matrices
	A  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
	B  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	C  = AllocateMatrix(B.height, B.width, 0);
    C_opt = AllocateMatrix(B.height, B.width, 0);

   /* Convolve matrix B with matrix A on the CPU. */
    Matrix reference = AllocateMatrix(C.height, C.width, 0);
    gettimeofday(&start1,NULL);
    computeGold(reference.elements, A.elements, B.elements, B.height, B.width);
    gettimeofday(&stop1,NULL);   
	
    /* Convolve matrix B with matrix A on the device. */
    float ktime,optktime;
    /* Un-optimized */
    gettimeofday(&start2,NULL);
    ktime = ConvolutionOnDevice(A, B, C);
    gettimeofday(&stop2,NULL);
    /* Optimized */
    gettimeofday(&start3,NULL);
    optktime = OptimizedConvolutionOnDevice(A,B,C_opt);
    gettimeofday(&stop3,NULL);    

    float cpuruntime = (float)(stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float)1000000);
    
    printf("CPU run time = %0.2f s\nDevice run time = %0.2f s || Speedup: %0.2f x\nOptimized device run time = %0.2f s || Speedup: %0.2f x\nParallel speedup: %0.2f x\n\n",cpuruntime,ktime,cpuruntime/ktime,optktime,cpuruntime/optktime,ktime/optktime);

   /* Check if the device result is equivalent to the expected solution. */
    int num_elements = C.height * C.width;
	int status = checkResults(reference.elements, C.elements, num_elements, 0.001f);
	printf("Test Unoptimized %s\n\n", (1 == status) ? "PASSED" : "FAILED");
    
    status = checkResults(reference.elements, C_opt.elements, num_elements, 0.001f);
    printf("Test Optimized %s\n", (1 == status) ? "PASSED" : "FAILED");

   // Free matrices
   FreeMatrix(&A);
   FreeMatrix(&B);
   FreeMatrix(&C);
   FreeMatrix(&C_opt);	
   return 0;
}

float ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    struct timeval start1, stop1, start2, stop2;

    gettimeofday(&start1,NULL);
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory

    // Setup the execution configuration
    dim3 dimBlock(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 dimGrid(N.width/dimBlock.x,N.height/dimBlock.y);

    gettimeofday(&start2,NULL);
    // Launch the device computation threads!
    ConvolutionKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd);
    cudaThreadSynchronize();

    gettimeofday(&stop2,NULL);
    
    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 
    
    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
    
    gettimeofday(&stop1,NULL);
    
    /* Timing calculations */
    float kernel_runtime = (float)(stop2.tv_sec - start2.tv_sec + (stop2.tv_usec - start2.tv_usec)/(float)1000000);
    float overhead = (float)(stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float)1000000) - kernel_runtime;

    printf("Unoptimized Kernel Runtime: %f\nOverhead: %f\nPercent Overhead: %f\n\n",kernel_runtime,overhead,(100*overhead)/(overhead + kernel_runtime));
    
    return kernel_runtime;
}

float OptimizedConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    struct timeval start1, stop1, start2, stop2;

    gettimeofday(&start1,NULL);
    // Load M and N to the device
    cudaMemcpyToSymbol(cM,M.elements,KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory

    // Setup the execution configuration
    dim3 dimBlock(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    dim3 dimGrid(N.width/dimBlock.x,N.height/dimBlock.y);

    gettimeofday(&start2,NULL);
    // Launch the device computation threads!
    OptimizedConvolutionKernel<<<dimGrid,dimBlock>>>(Nd,Pd);
    cudaThreadSynchronize();

    gettimeofday(&stop2,NULL);
    
    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 
    
    // Free device matrices
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
    
    gettimeofday(&stop1,NULL);
    
    /* Timing calculations */
    float kernel_runtime = (float)(stop2.tv_sec - start2.tv_sec + (stop2.tv_usec - start2.tv_usec)/(float)1000000);
    float overhead = (float)(stop1.tv_sec - start1.tv_sec + (stop1.tv_usec - start1.tv_usec)/(float)1000000) - kernel_runtime;
    
    printf("Optimized Kernel Runtime: %f\nOverhead: %f\nPercent Overhead: %f\n\n",kernel_runtime,overhead,(100*overhead)/(overhead + kernel_runtime));
    /* Return run time */
    return kernel_runtime;
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++){
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Check the CPU and GPU solutions
int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    int offBy = 0;
    for(int i = 0; i < num_elements; i++) {
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            offBy++;
            checkMark = 0;
        }
        if(PRINT_RESULTS)
        printf("%f : %f\n",reference[i],gpu_result[i]);
    }
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \nOff by: %d\n", epsilon, offBy); 
    return checkMark;
}
