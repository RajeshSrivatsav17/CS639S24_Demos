#pragma once

#include "Parameters.h"

void MatMatMultiply(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_K],
    const float (&B)[MATRIX_SIZE_K][MATRIX_SIZE_N], float (&C)[MATRIX_SIZE_M][MATRIX_SIZE_N]);

void MatMatMultiplyReference(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_K],
    const float (&B)[MATRIX_SIZE_K][MATRIX_SIZE_N], float (&C)[MATRIX_SIZE_M][MATRIX_SIZE_N]);