#include "Reductions.h"

#include <algorithm>
#define DO_NOT_USE_MKL
#ifndef DO_NOT_USE_MKL
#include <mkl.h>
#endif

float Norm(const float (&x)[XDIM][YDIM][ZDIM])
{
    float result = 0.;
#ifdef DO_NOT_USE_MKL
#pragma omp parallel for reduction(max:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        result = std::max(result, std::abs(x[i][j][k]));
#else
    const float* tmp = reinterpret_cast<const float*>(x);
    result = std::abs(tmp[cblas_isamax (XDIM * YDIM * ZDIM, &x[0][0][0], 1)]);
#endif
    return result;
}

float InnerProduct(const float (&x)[XDIM][YDIM][ZDIM], const float (&y)[XDIM][YDIM][ZDIM])
{
    double result = 0.;
//float cblas_sdot (const MKL_INT n, const float *x, const MKL_INT incx, const float *y, const MKL_INT incy);
#ifdef DO_NOT_USE_MKL    
#pragma omp parallel for reduction(+:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        result += (double) x[i][j][k] * (double) y[i][j][k];
#else
    result = cblas_sdot (
        XDIM * YDIM * ZDIM,
        &x[0][0][0], 
        1, 
        &y[0][0][0], 
        1);
#endif
    return (float) result;
}
