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

double** MatrixOps::multiply (
        double **matrix_one, const size_t &rows_one, const size_t &cols_one,
        double **matrix_two, const size_t &rows_two, const size_t &cols_two) {
    // We check for correct matrix sizes before allocating any memory.
    if (cols_one != rows_two) {
        //TODO This should be logged.
        throw logic_error("Matrices are not of compatible size.");
    }

    double **result = newMatrix(rows_one, cols_two);

    for (size_t r = 0; r < rows_one; r++) {
        for (size_t c = 0; c < cols_two; c++) {
            result[r][c] = 0;
            for (size_t i = 0; i < cols_one; i++) {
                result[r][c] += matrix_one[r][i] * matrix_two[i][c];
            }
        }
    }
    return result;
}

