#ifndef MT_H_
#define MT_H_
#include "chol.h"
/* Arguments struct for pthread arguments */
typedef struct {
    int thread_num;
    unsigned int k;
    Matrix* matrix;
} arguments;

#endif
