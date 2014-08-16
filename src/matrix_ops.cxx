#include "matrix_ops.hpp"

double** MatrixOps::transpose(double **matrix, const size_t rows, const size_t cols) {
    double **result = new double*[cols];
    for (int i = 0; i < cols; i++) {
        result[i] = new double[rows];
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            result[c][r] = matrix[r][c];
        }
    }
    return result;
}

