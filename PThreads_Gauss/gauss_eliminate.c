/* Gaussian elimination code.
 * Author: Naga Kandasamy
 * Date created: 02/07/2014
 * Date of last update: 01/30/2017
 * Compile as follows: gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -lpthread -std=c99 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"
#include "mt.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define NUM_THREADS 2 

typedef struct param_t {
    int thread_count;
    float* U;
    calc_t* calculations;
} param_t;

pthread_mutex_t mutex_lock;
int threads_remaining = NUM_THREADS;
int incr = 0;
int j,k,i;
int finished,no_instructions;
/* Function prototypes. */
extern int compute_gold (float *, unsigned int);
Matrix allocate_matrix (int num_rows, int num_columns, int init);
void gauss_eliminate_using_pthreads (Matrix);
int perform_simple_check (const Matrix);
void print_matrix (const Matrix);
float get_random_number (int, int);
int check_results (float *, float *, unsigned int, float);
void* compute_gold_p(void* args_in);

int
main (int argc, char **argv)
{
  /* Check command line arguments. */
  if (argc > 1)
    {
      printf ("Error. This program accepts no arguments. \n");
      exit (0);
    }

  /* Matrices for the program. */
  Matrix A;			// The input matrix
  Matrix U_reference;		// The upper triangular matrix computed by the reference code
  Matrix U_mt;			// The upper triangular matrix computed by the pthread code

  /* Initialize the random number generator with a seed value. */
  srand (time (NULL));

  /* Allocate memory and initialize the matrices. */
  A = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);	// Allocate and populate a random square matrix
  U_reference = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	// Allocate space for the reference result
  U_mt = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	// Allocate space for the multi-threaded result

  /* Copy the contents of the A matrix into the U matrices. */
  for (int i = 0; i < A.num_rows; i++)
    {
      for (int j = 0; j < A.num_rows; j++)
	{
	  U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
	  U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
	}
    }

  printf ("Performing gaussian elimination using the reference code. \n");
  struct timeval start, stop;
  gettimeofday (&start, NULL);
  int status = compute_gold (U_reference.elements, A.num_rows);
  gettimeofday (&stop, NULL);
  printf ("CPU run time = %0.2f s. \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));

  if (status == 0)
    {
      printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
      exit (0);
    }
  status = perform_simple_check (U_reference);	// Check that the principal diagonal elements are 1 
  if (status == 0)
    {
      printf ("The upper triangular matrix is incorrect. Exiting. \n");
      exit (0);
    }
  printf ("Single-threaded Gaussian elimination was successful. \n");
    struct timeval start2,stop2;
  /* Perform the Gaussian elimination using pthreads. The resulting upper triangular matrix should be returned in U_mt */
    gettimeofday(&start2,NULL);
  gauss_eliminate_using_pthreads (U_mt);
    gettimeofday(&stop2,NULL);

    printf ("CPU run time = %0.2f s. \n",
      (float) (stop2.tv_sec - start2.tv_sec +
           (stop2.tv_usec - start2.tv_usec) / (float) 1000000));

  /* check if the pthread result is equivalent to the expected solution within a specified tolerance. */
  int size = MATRIX_SIZE * MATRIX_SIZE;
  int res = check_results (U_reference.elements, U_mt.elements, size, 0.001f);
  printf ("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
//    printf("%f,%f",U_mt.elements,U_reference.elements);
  /* Free memory allocated for the matrices. */
  free (A.elements);
  free (U_reference.elements);
  free (U_mt.elements);

  return 0;
}


/* Write code to perform gaussian elimination using pthreads. */
void
gauss_eliminate_using_pthreads (Matrix U)
{
   pthread_t pthreads[NUM_THREADS];// = malloc(NUM_THREADS * sizeof(pthread_t));
   param_t* params;
    params = (param_t*) malloc(sizeof(param_t));
    params->U = (float*)U.elements;
    params->thread_count = 0;
    params->calculations = (calc_t*) malloc(sizeof(calc_t));
    params->calculations->i = -1;
    pthread_mutex_init(&mutex_lock,NULL);
    no_instructions = 0; 
    int i=0;
    int rv;

    for(i=0; i<NUM_THREADS; i++) {
        rv = (int) pthread_create(&pthreads[i], NULL,compute_gold_p,(void*) params);
        if(rv)
            printf("Error creating pthread");
    }
    
    for(i=0; i<NUM_THREADS; i++){
        rv = (int) pthread_join(pthreads[i],NULL);
        if(rv)
            printf("Error joining threads");
    }
    

    //free(pthreads);
    //pthread_exit(NULL); 
}

void* compute_gold_p(void* args_in) {
    
    param_t* arguments = args_in;
    while(pthread_mutex_trylock(&mutex_lock)!=0){
    }
    float* U = arguments->U;
    int thread_num = arguments->thread_count++;
    pthread_mutex_unlock(&mutex_lock);
    finished = 0;
    int num_elements = NUM_THREADS; 
    // First thread compiles all calculation attributes
    // and places them in calculation queue for other threads
    // calculate
    if(thread_num == 0) {
        calc_t* tail = arguments->calculations;
        int i,j,k;
        i=-2;
        int num_elements = MATRIX_SIZE;
        for(k=0; k< num_elements; k++) {
            for(j=(k+1); j < num_elements; j++) {
        //        while(pthread_mutex_trylock(&mutex_lock)!=0){
          //      }
                tail = push(tail,i,j,k,2);
            //    pthread_mutex_unlock(&mutex_lock);
            }

            //while(pthread_mutex_trylock(&mutex_lock)!=0){
            //}
            tail = push(tail,i,j,k,1);
            //pthread_mutex_unlock(&mutex_lock);

            for(i=(k+1); i< num_elements; i++) {
                for(j=(k+1); j<num_elements; j++){
               //     while(pthread_mutex_trylock(&mutex_lock)!=0){
                 //   }
                    tail = push(tail,i,j,k,3);
                   // pthread_mutex_unlock(&mutex_lock);
                }
                //while(pthread_mutex_trylock(&mutex_lock)!=0){
                //}   
                tail = push(tail,i,j,k,0);
               // pthread_mutex_unlock(&mutex_lock);
            }
        }
        puts("first thread done");
        finished = 1;
    }
    puts("here");
//    no_instructions = 0;
    while(!finished){

    }

    while(!finished || !no_instructions) {
        //while(pthread_mutex_trylock(&mutex_lock)!=0){
        //}
            
            calc_t* current_instruction = arguments->calculations;
            if(current_instruction != NULL)
                arguments->calculations = arguments->calculations->next_node;
         //   pthread_mutex_unlock(&mutex_lock);
        
        if(current_instruction == NULL || no_instructions == 1) {
            no_instructions = 1;
            //puts("no instruction");
        //    pthread_mutex_unlock(&mutex_lock);
            continue;
        }
        /*else {
            no_instructions = 0;
            //puts("calcing with instr");
        }*/
        if(current_instruction == 0) {
            //while(pthread_mutex_trylock(&mutex_lock)!=0){
            //}
                U[(num_elements * current_instruction->i)+current_instruction->k] = 1;
            //pthread_mutex_unlock(&mutex_lock);
        }
        else if(current_instruction->type == 1) {
            //while(pthread_mutex_trylock(&mutex_lock)!=0){
            //}
               U[(current_instruction->k)*(num_elements + 1)] = 1; 
            //pthread_mutex_unlock(&mutex_lock);
        }
        else if(current_instruction->type == 2) {
            //while(pthread_mutex_trylock(&mutex_lock)!=0){
            //}
            
            U[(num_elements * current_instruction->k) + current_instruction->j] = (float) (U[num_elements * current_instruction->k + current_instruction->j] / U[num_elements * current_instruction->k + current_instruction->k]);
            
        //    pthread_mutex_unlock(&mutex_lock);
        }
        else if(current_instruction->type == 3) {
          //  while(pthread_mutex_trylock(&mutex_lock)!=0){
            //}
            
            U[num_elements * current_instruction->i + current_instruction->j] = U[num_elements * current_instruction->i + current_instruction->j] - (U[num_elements * current_instruction->i + current_instruction->k] * U[num_elements * current_instruction->k + current_instruction->j]);    // Elimination step

            
            //pthread_mutex_unlock(&mutex_lock);
        }
        //pthread_mutex_unlock(&mutex_lock);
    }
    pthread_exit(NULL);
}

/* Function checks if the results generated by the single threaded and multi threaded versions match. */
        int
check_results (float *A, float *B, unsigned int size, float tolerance)
{
        for (int i = 0; i < size; i++)
                if (fabsf (A[i] - B[i]) > tolerance)
                    return 0;
        return 1;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
 */
        Matrix
allocate_matrix (int num_rows, int num_columns, int init)
{
        Matrix M;
        M.num_columns = M.pitch = num_columns;
        M.num_rows = num_rows;
        int size = M.num_rows * M.num_columns;

        M.elements = (float *) malloc (size * sizeof (float));
        for (unsigned int i = 0; i < size; i++)
        {
                if (init == 0)
                        M.elements[i] = 0;
                else
                        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
        }
        return M;
}


/* Returns a random floating-point number between the specified min and max values. */ 
        float
get_random_number (int min, int max)
{
        return (float)
                floor ((double)
                                (min + (max - min + 1) * ((float) rand () / (float) RAND_MAX)));
}

/* Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1. */
        int
perform_simple_check (const Matrix M)
{
        for (unsigned int i = 0; i < M.num_rows; i++)
                if ((fabs (M.elements[M.num_rows * i + i] - 1.0)) > 0.001)
                        return 0;
        return 1;
}
