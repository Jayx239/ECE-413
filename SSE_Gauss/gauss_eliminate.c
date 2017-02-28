/* Gaussian elimination code.
 * Author: Naga Kandasamy, 10/24/2015
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -std=c99 -O3 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

extern int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void gauss_eliminate_using_sse(const Matrix, Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, unsigned int, float);


int 
main(int argc, char** argv) {
    if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}	

    /* Allocate and initialize the matrices. */
	Matrix  A;                                              /* The N x N input matrix. */
	Matrix  U;                                              /* The upper triangular matrix to be computed. */
	
	srand(time(NULL));
		
    A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);      /* Create a random N x N matrix. */
	U  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);      /* Create a random N x 1 vector. */
		
	/* Gaussian elimination using the reference code. */
	Matrix reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	struct timeval start, stop, start2, stop2;	
	gettimeofday(&start, NULL);

	printf("Performing gaussian elimination using the reference code. \n");
	int status = compute_gold(reference.elements, A.elements, A.num_rows);

	gettimeofday(&stop, NULL);
	printf("CPU run time = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	if(status == 0){
		printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
		exit(0);
	}
	status = perform_simple_check(reference); // Check that the principal diagonal elements are 1 
	if(status == 0){
		printf("The upper triangular matrix is incorrect. Exiting. \n");
		exit(0); 
	}
	printf("Gaussian elimination using the reference code was successful. \n");

	/* WRITE THIS CODE: Perform the Gaussian elimination using the SSE version. 
     * The resulting upper triangular matrix should be returned in U
     * */
    gettimeofday(&start2, NULL);
	gauss_eliminate_using_sse(A, U);
    gettimeofday(&stop2, NULL);
	/* Check if the SSE result is equivalent to the expected solution. */

    printf("SSE run time = %0.2f s. \n", (float)(stop2.tv_sec - start2.tv_sec + (stop2.tv_usec - start2.tv_usec)/(float)1000000));

	int size = MATRIX_SIZE*MATRIX_SIZE;
	int res = check_results(reference.elements, U.elements, size, 0.001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	free(A.elements); A.elements = NULL;
	free(U.elements); U.elements = NULL;
	free(reference.elements); reference.elements = NULL;

	return 0;
}


void 
gauss_eliminate_using_sse(const Matrix A, Matrix U)                  /* Write code to perform gaussian elimination using OpenMP. */
{
    /* Copy input matrix */
    //memcpy(&U,&A,sizeof(A));
    float* u_elems = U.elements;//malloc(sizeof(A.elements)+15);
     
    unsigned int i,j,k,z; 
    int r_num_rows = A.num_rows;
    int num_rows = r_num_rows; 
    __m128 iptr,jptr,kptr,inkptr,temp,temp2;
    
    for(i=0; i<num_rows/4; i++) {
        for(j=0; j<num_rows/4; j++) {
            jptr = _mm_load_ps(&A.elements[((num_rows*i)+j)*4]);
            _mm_store_ps(&u_elems[((num_rows*i)+j*4)],jptr);
        }
    }    

    for(k=0; k<num_rows/4; k++)
    {
        inkptr = _mm_load_ps1(&u_elems[((num_rows*k) + k)]);//kptr+(k*(num_rows/4));
//        jptr = kptr+(num_rows/4) ;
        
        for(j=(k+1); j<(num_rows/4); j++)
        {
            printf("k:%d   j:%d\n",k,j);
            if (u_elems[((num_rows*k) + k)] == 0){
                printf("Numerical instability detected. The principal diagonal element is zero. \n");
                return;
            }

            inkptr = _mm_load_ps1(&u_elems[((num_rows*k) + k)*4]); 
            jptr = _mm_load_ps(&u_elems[((num_rows*k)+j)*4]);
            temp = _mm_div_ps(jptr,inkptr);
            //jptr = temp;
            _mm_store_ps(&u_elems[((num_rows*k)+j*4)],temp);
            //jptr++;
        }
        
        for(i=k*4; i>(k*4)-4; i--)
        {
            if(i < 0)
                break;
            u_elems[((num_rows*i)+i)] = 1.0f;
            //u_elems[((num_rows*(k+1))+k+1)] = 1.0f;
            //u_elems[((num_rows*(k+2))+k+2)] = 1.0f;
            //u_elems[((num_rows*(k+3))+k+3)] = 1.0f;
        }
//        printf("Set princ to 1\n");
        
        for(i=(k+1); i< (num_rows/4); i++)
        {
            iptr = _mm_load1_ps(&u_elems[((num_rows*i)+k)]);
//            jptr = _mm_load1_ps(&U.elements[num_rows*i+j];
            //iptr = &vector_u[num_rows*i+k];
            //jptr = iptr+(num_rows/4); 
            //inkptr = &vector_u[((num_rows/4)*k)+k];
            for(j=(k+1); j<(num_rows/4); j++)
            {
                  //  printf("k:%d    j:%d\n",k,j);
                
                iptr = _mm_load1_ps(&u_elems[((num_rows*i)+k)*4]);
                jptr = _mm_load_ps(&u_elems[((num_rows*i)+j)*4]);
                inkptr = _mm_load_ps(&u_elems[((num_rows*k) + j)*4]);
                
                temp = _mm_mul_ps(iptr,inkptr);
                jptr = _mm_sub_ps(jptr,temp);
                _mm_store_ps(&u_elems[((num_rows*i)+j)*4],jptr); 
/*
                iptr = _mm_load1_ps(&u_elems[((num_rows*i)+k)]);
                jptr = _mm_load_ps(&u_elems[((num_rows*(i))+j)*4]);
                inkptr = _mm_load_ps(&u_elems[((num_rows*k) + j*4)]);

                temp = _mm_mul_ps(iptr,inkptr);
                jptr = _mm_sub_ps(jptr,temp);
                _mm_store_ps(&u_elems[((num_rows*i)+j*4)],jptr);
  */              
            }
    
        //for(z=i*4; z>(i*4)-4; z--)
          //  {
            //if(i < 0)
              //  break;
                u_elems[((num_rows*i)+k)] = 0.0f;       
            //}
            //for(j=i*4; j<(i*4)+4; j++)
                //u_elems[((num_rows*j)+k)] = 0;

            //printf("Made it to end of loop\n"); 
        }
    }
//    memcpy(U.elements,u_elems,sizeof(U.elements));
}


int 
check_results(float *A, float *B, unsigned int size, float tolerance)   /* Check if refernce results match multi threaded results. */
{
    int num_off = 0;
    int num_miss_zeros = 0;
    int num_hit_zeros = 0;
    int num_hit_ones = 0;
    int expected_1s = 0;
    int expected_0s = 0;

	for(int i = 0; i < size; i++){
        if(A[i] == 1)
            expected_1s++;
        if(A[i] == 0)
            expected_0s++;
		if(fabsf(A[i] - B[i]) > tolerance){
            num_off++;    
    //        printf("Expected:%f   Actual: %f\n",A[i],B[i]); 
            if(A[i] == 0)
                num_miss_zeros++;
        }
        else if(A[i] == 0)
            num_hit_zeros++;
        else if(A[i] == 1)
            num_hit_ones++;
    }
    printf("Percent difference: %f\nPercentage of non-zeros: %f\nPercentage hit 0: %f\n Percentage hit 1: %f\n",num_off/(size*1.0),num_miss_zeros/(size*1.0),num_hit_zeros/(expected_0s*1.0),num_hit_ones/(expected_1s*1.0));

    if(num_off != 0)
        return 0;
	
    return 1;
}


/* Allocate a matrix of dimensions height*width. 
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization.
 * */
Matrix 
allocate_matrix(int num_rows, int num_columns, int init){
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
	M.elements = (float*) malloc(size*sizeof(float));
	
    for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}

    return M;
}	


float 
get_random_number(int min, int max){                                    /* Returns a random FP number between min and max values. */
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

int 
perform_simple_check(const Matrix M){                                   /* Check for upper triangular matrix, that is, the principal diagonal elements are 1. */
    int num_off = 0;
    for(unsigned int i = 0; i < M.num_rows; i++)
        if((fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) 
            num_off++;

    if(num_off != 0)
        return 0;
	
    return 1;
} 


