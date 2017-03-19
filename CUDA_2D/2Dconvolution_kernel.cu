
#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    double sum = 0;
    
    unsigned int mBeg = i < 2 ? 2-i: 0;
    unsigned int mEnd = i>(N.height-3) ? N.height-i+2  : 5;
    unsigned int nBeg = j < 2 ? 2-j : 0;
    unsigned int nEnd = j > (N.width-3) ? N.width-j+2  : 5;

    for(int m=mBeg; m < mEnd; m++)
        for(int n=nBeg; n < nEnd; n++)     
            sum += M.elements[M.width*m + n] * N.elements[N.width*(i + m-2) + ((j + n-2))];
        
    P.elements[ N.width*i + j] = (float)sum;

}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
