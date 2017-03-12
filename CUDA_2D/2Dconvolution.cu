#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// includes, kernels
#include "2Dconvolution_kernel.cu"

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
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
int checkResults(float *, float *, int, float);


int main(int argc, char** argv) 
{

	Matrix  A;
	Matrix  B;
	Matrix  C;
	
	srand(time(NULL));
	
	// Allocate and initialize the matrices
	A  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
	B  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	C  = AllocateMatrix(B.height, B.width, 0);

    
   /* Convolve matrix B with matrix A on the CPU. */
   Matrix reference = AllocateMatrix(C.height, C.width, 0);
   computeGold(reference.elements, A.elements, B.elements, B.height, B.width);
       
	/* Convolve matrix B with matrix A on the device. */
   ConvolutionOnDevice(A, B, C);

   /* Check if the device result is equivalent to the expected solution. */
    int num_elements = C.height * C.width;
	int status = checkResults(reference.elements, C.elements, num_elements, 0.001f);
	printf("Test %s\n", (1 == status) ? "PASSED" : "FAILED");

   // Free matrices
   FreeMatrix(&A);
   FreeMatrix(&B);
   FreeMatrix(&C);
	
   return 0;
}


void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory

    // Setup the execution configuration


 
    // Launch the device computation threads!

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);

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
    
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}
