#include "matrix_ops.hpp"

double** MatrixOps::newMatrix(const size_t &rows, const size_t &cols) {
    double **result = new double*[rows];
    for (size_t i = 0; i < rows; i++) {
        result[i] = new double[cols];
    }
    return result;
}

void MatrixOps::deleteMatrix(double **&matrix, const size_t &rows) {
    for (size_t i = 0; i < rows; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
    matrix = NULL;
}

double** MatrixOps::transpose(double **matrix, const size_t &rows, const size_t &cols) {
    double **result = newMatrix(cols, rows);

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            result[c][r] = matrix[r][c];
        }
    }
    return result;
}

double** MatrixOps::add(double **matrix_one, double **matrix_two, const size_t &rows, const size_t &cols) {
    double **result = newMatrix(rows, cols);

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            result[r][c] = matrix_one[r][c] + matrix_two[r][c];
        }
    }
    return result;
}

double** MatrixOps::multiply(double **matrix, const double &scalar, const size_t &rows, const size_t &cols) {
    double **result = newMatrix(rows, cols);

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            result[r][c] = matrix[r][c] * scalar;
        }
    }
    return result;
}

