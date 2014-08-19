#include <cassert>
#include <iostream>
#include "ann.hpp"
#include "log_sigmoid.hpp"
#include "matrix_ops.hpp"

template <typename K>
Ann<K>::Ann(K kernel) {
    this->kernel = kernel;
}

//TODO Invalid write happens in here?
template <typename K>
double** Ann<K>::feedForward(
        double **input, const size_t &input_rows, const size_t &input_cols,
        double **weights, const size_t &weights_rows, const size_t &weights_cols,
        double **bias, const size_t &bias_rows, const size_t &bias_cols) {
    assert(bias_cols == 1);
    assert(input_rows == bias_rows);
    double **input_bias = MatrixOps::horizontalConcat(input, input_rows, input_cols, bias, bias_rows, bias_cols);
    double **net = MatrixOps::multiply(weights, weights_rows, weights_cols, input_bias, weights_rows, weights_cols + bias_cols);
    cout << "New Height: " << weights_rows << endl;
    cout << "New Width: " << weights_cols + bias_cols << endl;
    double **result = MatrixOps:: newMatrix(input_rows, input_cols + bias_cols);
    for (size_t r = 0; r < input_rows; r++) {
        for (size_t c = 0; c < input_cols + bias_cols; c++) {
            result[r][c] = kernel.kernelFunc(net[r][c]);
        }
    }
    return result;
}

// List all Kernels used here.
template class Ann<LogSigmoid>;

