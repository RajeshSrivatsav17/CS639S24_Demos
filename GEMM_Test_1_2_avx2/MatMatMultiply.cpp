#include "MatMatMultiply.h"
#include "MatMatMultiplyBlockHelper.h"
#include "mkl.h"

alignas(64) float localA[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localB[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localC[BLOCK_SIZE][BLOCK_SIZE];

#pragma omp threadprivate(localA, localB, localC)

void MatMatMultiply(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_K],
    const float (&B)[MATRIX_SIZE_K][MATRIX_SIZE_N], float (&C)[MATRIX_SIZE_M][MATRIX_SIZE_N])
{
    static constexpr int NBLOCKS_I = MATRIX_SIZE_M / BLOCK_SIZE;
    static constexpr int NBLOCKS_J = MATRIX_SIZE_N / BLOCK_SIZE;
    static constexpr int NBLOCKS_K = MATRIX_SIZE_K / BLOCK_SIZE;

  //  Mx k x k x N
    using blocked_matrix_t_C = float (&) [NBLOCKS_I][BLOCK_SIZE][NBLOCKS_J][BLOCK_SIZE];
    using const_blocked_matrix_t_A = const float (&) [NBLOCKS_I][BLOCK_SIZE][NBLOCKS_K][BLOCK_SIZE];
    using const_blocked_matrix_t_B = const float (&) [NBLOCKS_K][BLOCK_SIZE][NBLOCKS_J][BLOCK_SIZE];

    auto blockA = reinterpret_cast<const_blocked_matrix_t_A>(A[0][0]);
    auto blockB = reinterpret_cast<const_blocked_matrix_t_B>(B[0][0]);
    auto blockC = reinterpret_cast<blocked_matrix_t_C>(C[0][0]);

#pragma omp parallel for
    for (int bi = 0; bi < NBLOCKS_I; bi++)
    for (int bj = 0; bj < NBLOCKS_J; bj++) {
        
        for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localC[ii][jj] = 0.;
            }

        for (int bk = 0; bk < NBLOCKS_K; bk++) { 

            for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localA[ii][jj] = blockA[bi][ii][bk][jj];
                localB[ii][jj] = blockB[bk][ii][bj][jj];
            }

            MatMatMultiplyBlockHelper(localA, localB, localC);
        }

        for (int ii = 0; ii < BLOCK_SIZE; ii++)
        for (int jj = 0; jj < BLOCK_SIZE; jj++)                
            blockC[bi][ii][bj][jj] = localC[ii][jj];
    }
}

void MatMatMultiplyReference(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_K],
    const float (&B)[MATRIX_SIZE_K][MATRIX_SIZE_N], float (&C)[MATRIX_SIZE_M][MATRIX_SIZE_N])
{
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        MATRIX_SIZE_M,
        MATRIX_SIZE_N,
        MATRIX_SIZE_K,
        1.,
        &A[0][0],
        MATRIX_SIZE_K,
        &B[0][0],
        MATRIX_SIZE_N,
        0.,
        &C[0][0],
        MATRIX_SIZE_N
    );
}
