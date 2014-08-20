#include "matrix_ops.hpp"

double** MatrixOps::newMatrix(const size_t &rows, const size_t &cols) {
    double **result = new double*[rows];
    for (size_t r = 0; r < rows; r++) {
        result[r] = new double[cols];
    }
    return result;
}

void MatrixOps::deleteMatrix(double **&matrix, const size_t &rows) {
    for (size_t r = 0; r < rows; r++) {
        delete[] matrix[r];
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

double** MatrixOps::scalarMultiply(double **matrix, const double &scalar, const size_t &rows, const size_t &cols) {
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

double** MatrixOps::hadamardMultiply(double **matrix_one, double **matrix_two, const size_t &rows, const size_t &cols) {
    double **result = newMatrix(rows, cols);

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            result[r][c] = matrix_one[r][c] * matrix_two[r][c];
        }
    }
    return result;
}

double** MatrixOps::kroneckerMultiply(
        double **matrix_one, const size_t &rows_one, const size_t &cols_one,
        double **matrix_two, const size_t &rows_two, const size_t &cols_two) {
    double **result = newMatrix(rows_one * rows_two, cols_one * cols_two);

    for (size_t r1 = 0; r1 < rows_one; r1++) {
        for (size_t c1 = 0; c1 < cols_one; c1++) {
            //TODO Extract matrix_one[r1][c1] to variable to speed up loop
            for (size_t r2 = 0; r2 < rows_two; r2++) {
                size_t currentRow = (r1 * rows_two) + r2;
                for (size_t c2 = 0; c2 < cols_two; c2++) {
                    size_t currentCol = (c1 * cols_two) + c2;
                    result[currentRow][currentCol] = matrix_one[r1][c1] * matrix_two[r2][c2];
                }
            }
        }
    }
    return result;
}

double** MatrixOps::horizontalConcat(
        double **matrix_one, const size_t &rows_one, const size_t &cols_one,
        double **matrix_two, const size_t &rows_two, const size_t &cols_two) {
    // We check for correct matrix sizes before allocating any memory.
    if (rows_one != rows_two) {
        //TODO This should be logged.
        throw logic_error("Matrices are not of compatible size.");
    }
    double **result = newMatrix(rows_one, cols_one + cols_two);

    for (size_t r = 0; r < rows_one; r++) {
        for (size_t c = 0; c < cols_one + cols_two; c++) {
            if (c < cols_one) {
                result[r][c] = matrix_one[r][c];
            } else {
                result[r][c] = matrix_two[r][c - cols_one];
            }
        }
    }
    return result;
}

double** MatrixOps::verticalConcat(
        double **matrix_one, const size_t &rows_one, const size_t &cols_one,
        double **matrix_two, const size_t &rows_two, const size_t &cols_two) {
    // We check for correct matrix sizes before allocating any memory.
    if (cols_one != cols_two) {
        //TODO This should be logged.
        throw logic_error("Matrices are not of compatible size.");
    }
    double **result = newMatrix(rows_one + rows_two, cols_one);

    for (size_t r = 0; r < rows_one + cols_one; r++) {
        for (size_t c = 0; c < cols_one; c++) {
            if (r < rows_one) {
                result[r][c] = matrix_one[r][c];
            } else {
                result[r][c] = matrix_two[r - rows_one][c];
            }
        }
    }
    return result;
}

unsigned int* MatrixOps::matrixToClass(double **matrix, const size_t &rows, const size_t &cols) {
    unsigned int *result = new unsigned int[rows];

    double **classes = new double*[rows];
    for (size_t r = 0; r < rows; r++) {
        classes[r] = NULL;
    }

    size_t next_class = 0;
    for (size_t matrix_row = 0; matrix_row < rows; matrix_row++) {
        long found_index = rowInMatrix(matrix[matrix_row], classes, rows, cols);
        if (found_index >= 0) {
            result[matrix_row] = found_index+1;
        } else {
            classes[next_class] = matrix[matrix_row];
            result[matrix_row] = next_class+1;
            next_class++;
        }
    }
    delete[] classes;
    return result;
}

long MatrixOps::rowInMatrix(double *row, double **matrix, const size_t &rows, const size_t &cols) {
    for (size_t r = 0; r < rows; r++) {
        bool is_equal = true;
        if (matrix[r] != NULL) {
            for (size_t c = 0; c < cols; c++) {
                if (matrix[r][c] != row[c]) {
                    is_equal = false;
                    break;
                }
            }
        } else {
            is_equal = false;
        }
        if (is_equal) {
            return r;
        }
    }
    return -1;
}

