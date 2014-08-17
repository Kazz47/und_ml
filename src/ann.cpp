#include <cassert>
#include <iostream>
#include "ann.hpp"
#include "matrix_ops.hpp"

unsigned int* Ann::matrixToClass(double **matrix, const size_t &rows, const size_t &cols) {
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

long Ann::rowInMatrix(double *row, double **matrix, const size_t &rows, const size_t &cols) {
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

