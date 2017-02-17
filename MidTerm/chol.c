/* Cholesky decomposition.
 * Compile as follows:
 * 						gcc -fopenmp -o chol chol.c chol_gold.c -lpthread -lm -std=c99
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "chol.h"
#include "mt.h"
#include <pthread.h>
#define NUM_THREADS 4
#define DEBUG 0

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

Matrix allocate_matrix(int num_rows, int num_columns, int init);
int perform_simple_check(const Matrix M);
void print_matrix(const Matrix M);
extern Matrix create_positive_definite_matrix(unsigned int, unsigned int);
extern int chol_gold(const Matrix, Matrix);
extern int check_chol(const Matrix, const Matrix);
void chol_using_pthreads(const Matrix, Matrix);
void chol_using_openmp(const Matrix, Matrix);

/* Helper Functions */
int get_turn();
void reset_turn(int size);
void decrement_turn();
void increment_turn();

pthread_mutex_t mutex;
volatile int* init_turn;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Matrices for the program
	Matrix A; // The N x N input matrix
	Matrix reference; // The upper triangular matrix computed by the CPU
	Matrix U_pthreads; // The upper triangular matrix computed by the pthread implementation
	Matrix U_openmp; // The upper triangular matrix computed by the openmp implementation 
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));

	// Create the positive definite matrix. May require a few tries if we are unlucky
	int success = 0;
	while(!success){
		A = create_positive_definite_matrix(MATRIX_SIZE, MATRIX_SIZE);
		if(A.elements != NULL)
				  success = 1;
	}
	// print_matrix(A);
	// getchar();


	reference  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the CPU result
	U_pthreads =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the pthread result
	U_openmp =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the openmp result


	// compute the Cholesky decomposition on the CPU; single threaded version	
	printf("Performing Cholesky decomposition on the CPU using the single-threaded version. \n");
	int status = chol_gold(A, reference);
	if(status == 0){
			  printf("Cholesky decomposition failed. The input matrix is not positive definite. \n");
			  exit(0);
	}
	
	
	printf("Double checking for correctness by recovering the original matrix. \n");
	if(check_chol(A, reference) == 0){
		printf("Error performing Cholesky decomposition on the CPU. Try again. Exiting. \n");
		exit(0);
	}
	printf("Cholesky decomposition on the CPU was successful. \n");

	/* MODIFY THIS CODE: Perform the Cholesky decomposition using pthreads. The resulting upper triangular matrix should be returned in 
	 U_pthreads */
	chol_using_pthreads(A, U_pthreads);

	/* MODIFY THIS CODE: Perform the Cholesky decomposition using openmp. The resulting upper traingular matrix should be returned in U_openmp */
	chol_using_openmp(A, U_openmp);


	// Check if the pthread and openmp results are equivalent to the expected solution
	if(check_chol(A, U_pthreads) == 0) 
			  printf("Error performing Cholesky decomposition using pthreads. \n");
	else
			  printf("Cholesky decomposition using pthreads was successful. \n");

	if(check_chol(A, U_openmp) == 0) 
			  printf("Error performing Cholesky decomposition using openmp. \n");
	else	
			  printf("Cholesky decomposition using openmp was successful. \n");



	// Free host matrices
	free(A.elements); 	
	free(U_pthreads.elements);	
	free(U_openmp.elements);
	free(reference.elements); 
	return 1;
}

void chol_thread_calc(void* input)
{
    /*Get input arguments */
    arguments* args = (arguments*) input;

    if(DEBUG)
        puts("Entered division step");
    int i,j,thread_num;
    thread_num = args->thread_num;

  /* Initialize values */
    while(get_turn() != (thread_num))
    {
        if(DEBUG)
            printf("Thread %d: Thread turn: %d\n",thread_num,get_turn());
    }

    float* U = args->matrix->elements;
    int k = args->k;
    unsigned int num_elements = args->matrix->num_rows;
  
    decrement_turn();

    /* Synchronize */
    while(get_turn() >= 0)
    {
        if(DEBUG)
            printf("Turn: %d\n",get_turn());
    
    }
    
    /* Set chunck size */
    int chunck_start = ((int) (num_elements/NUM_THREADS)) * thread_num;
    int end_chunck = chunck_start + ((int)(num_elements/NUM_THREADS));
    
    /* Ignore earlier */
    if(chunck_start < k+1)
        chunck_start = k+1;
    
    /* make last thread go until end*/
    if( thread_num == NUM_THREADS-1)
        end_chunck = num_elements;
  
    if(DEBUG)
        printf("Starting loop: Thread %d\n",thread_num); 

    /* Divsion Step */
    for(j = chunck_start; j < end_chunck; j++)
        U[k * num_elements + j] /= U[k * num_elements + k]; // Division step
    
    decrement_turn();
  
    /* Synchronize */ 
    while(get_turn() >= -NUM_THREADS && get_turn() != (NUM_THREADS-1))
    {
        if(DEBUG)
            printf("Waiting in turn: %d\n",get_turn());
    }  

    /* Elimination Step */
    for(i = (k + 1 + thread_num); i < num_elements; i+=NUM_THREADS)
        for(j = i; j < num_elements; j++)
                                    U[i * num_elements + j] -= U[k * num_elements + i] * U[k * num_elements + j];

    /* Reset counter */
    reset_turn(NUM_THREADS-1);

}

/* Write code to perform Cholesky decopmposition using pthreads. */
void chol_using_pthreads(const Matrix A, Matrix U)
{
    /* Variables */
    unsigned int i,j,k;
    int rv;
    pthread_t pthreads[NUM_THREADS];
    arguments args[NUM_THREADS];
 
    /* Copy contents of original matrix */
    for(i=0; i<A.num_rows; i++)
        U.elements[i] = A.elements[i];

    /* Initialize */ 
    pthread_mutex_init(&mutex,NULL);
    reset_turn((NUM_THREADS-1));

    for(j=0; j<NUM_THREADS; j++)
    {
        args[j].thread_num = j;
        args[j].matrix = &U; 
    }

    /* Perform elimination */
    for(k=0; k<U.num_rows; k++)
    {

            // Take the square root of the diagonal element
            U.elements[k * U.num_rows + k] = sqrt(U.elements[k * U.num_rows + k]);
            if(U.elements[k * U.num_rows + k] <= 0){
                    printf("Cholesky decomposition failed. \n");
                    return 0;
            }    

            /* Create threads */
            for(j=0; j<NUM_THREADS; j++)
            {
                    args[j].k = k;
                    rv = (int) pthread_create(&pthreads[j], NULL, chol_thread_calc,(void*) &args[j]);

                    if(rv)
                            printf("Error creating thread: %d",j);
            }

            /* Join threads */
            for(j=0; j<NUM_THREADS; j++)
            {
                    rv = (int) pthread_join(pthreads[j],NULL);
                    if(rv)
                            printf("Error joining thread: %d\n",j);
            }  
    
    }

    /* Final Step - zero out bottom triangle portion */
    for(i = 0; i < U.num_rows; i++)
        for(j = 0; j < i; j++)
            U.elements[i * U.num_rows + j] = 0.0;
    
    if(DEBUG) 
        print_matrix(U);

}

/* Write code to perform Cholesky decopmposition using openmp. */
void chol_using_openmp(const Matrix A, Matrix U)
{
}


// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
        Matrix M;
        M.num_columns = M.pitch = num_columns;
        M.num_rows = num_rows;
        int size = M.num_rows * M.num_columns;

        M.elements = (float *) malloc(size * sizeof(float));
        for(unsigned int i = 0; i < size; i++){
                if(init == 0) M.elements[i] = 0; 
                else
                        M.elements[i] = (float)rand()/(float)RAND_MAX;
        }
        return M;
}	

/* Helper functions */
int get_turn()
{
  if(DEBUG)
    puts("in get turn");
  while(pthread_mutex_trylock(&mutex)!=0){}; 
  int turn = *init_turn;pthread_mutex_unlock(&mutex); 
  if(DEBUG)
    puts("out get turn");
  return turn;
}

void reset_turn(int size)
{
  if(DEBUG)
    puts("in reset turn");
  while(pthread_mutex_trylock(&mutex)!=0)
  {
  }

  if(init_turn == NULL)
    init_turn = malloc(sizeof(int));

  *init_turn = size;
  pthread_mutex_unlock(&mutex); 
  if(DEBUG)
    puts("out reset turn");
}

void decrement_turn()
{
  if(DEBUG)
    puts("in decrement");
  while(pthread_mutex_trylock(&mutex)!=0)
  { 
  }
  
  *init_turn= *init_turn-1;
  pthread_mutex_unlock(&mutex);
  if(DEBUG)
    puts("out decrement");
}
void increment_turn()
{
  if(DEBUG)
    puts("in increment"); 
  while(pthread_mutex_trylock(&mutex)!=0)
  {
  }
  
  *init_turn= *init_turn-1; 
  pthread_mutex_unlock(&mutex); 
  if(DEBUG)
    puts("out increment");
}

/* End helper functions */


