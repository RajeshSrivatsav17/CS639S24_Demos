#pragma once

#include "Parameters.h"

#include <cstdlib>

void* AlignedAllocate(const std::size_t size, const std::size_t alignment);
void InitializeMatrices(float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_K],float (&B)[MATRIX_SIZE_K][MATRIX_SIZE_N]);
float MatrixMaxDifference(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_N],const float (&B)[MATRIX_SIZE_M][MATRIX_SIZE_N]);
