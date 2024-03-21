#include "ConjugateGradients.h"
#include "Laplacian.h"
#include "PointwiseOps.h"
#include "Reductions.h"
#include "Utilities.h"
#include "Timer.h"

#include <iostream>

extern Timer timerCG;

void ConjugateGradients(
    CSRMatrix& matrix,
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    const bool writeIterations)
{
    // Algorithm : Line 2
    timerCG.Restart(); ComputeLaplacian(matrix, x, z); 
    Saxpy(z, f, r, -1);
    float nu = Norm(r); timerCG.Pause();

    // Algorithm : Line 3
    if (nu < nuMax) return;
        
    // Algorithm : Line 4
    timerCG.Restart(); Copy(r, p);

    float rho=InnerProduct(p, r); timerCG.Pause();
        
    // Beginning of loop from Line 5
    for(int k=0;;k++)
    {
        std::cout << "Residual norm (nu) after " << k << " iterations = " << nu << std::endl;

        // Algorithm : Line 6
        timerCG.Restart(); ComputeLaplacian(matrix, p, z);
        float sigma=InnerProduct(p, z); timerCG.Pause();

        // Algorithm : Line 7
        float alpha=rho/sigma;

        // Algorithm : Line 8
        //Saxpy(z, r, r, -alpha);
        timerCG.Restart(); Saxpy(z, r, -alpha); // Calling the new Saxpy function
        nu=Norm(r); timerCG.Pause();

        // Algorithm : Lines 9-12
        if (nu < nuMax || k == kMax) {
            timerCG.Restart(); Saxpy(p, x, alpha); timerCG.Pause();
            //Saxpy(p, x, x, alpha);// Calling the new Saxpy function

            std::cout << "Conjugate Gradients terminated after " << k << " iterations; residual norm (nu) = " << nu << std::endl;
            if (writeIterations) WriteAsImage("x", x, k, 0, 127);
            return;
        }
            
        // Algorithm : Line 13
        timerCG.Restart(); Copy(r, z);
        float rho_new = InnerProduct(z, r); timerCG.Pause();

        // Algorithm : Line 14
        float beta = rho_new/rho;

        // Algorithm : Line 15
        rho=rho_new;

        timerCG.Restart(); 
        // Algorithm : Line 16
        Saxpy(p, x, alpha);
        // Note: this used to be 
        // Saxpy(p, x, x, alpha);
        // The version above uses the fact that the destination vector is the same
        // as the second input vector -- i.e. Saxpy(x, y, c) performs
        // the operation y += c * x
        Saxpy(p, r, p, beta); timerCG.Pause();

        if (writeIterations) WriteAsImage("x", x, k, 0, 127);
    }

}
