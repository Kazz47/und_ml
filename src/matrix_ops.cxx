#include "matrix_ops.hpp"

double** MatrixOps::newMatrix(const size_t &rows, const size_t &cols) {
    double **result = new double*[rows];
    for (int i = 0; i < rows; i++) {
        result[i] = new double[cols];
    }
    return result;
}

void MatrixOps::deleteMatrix(double **&matrix, const size_t &rows) {
    for (int i = 0; i < rows; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
    matrix = NULL;
}

double** MatrixOps::transpose(double **&matrix, const size_t &rows, const size_t &cols) {
    double **result = newMatrix(cols, rows);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            result[c][r] = matrix[r][c];
        }
    }
    return result;
}

