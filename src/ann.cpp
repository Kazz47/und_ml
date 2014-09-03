#include <cassert>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include "ann.hpp"
#include "log_sigmoid.hpp"
#include "matrix_ops.hpp"

template <typename K>
Ann<K>::Ann(K kernel) {
    this->kernel = kernel;
    //TODO This should log the seed every time and there should be another
    //constructor that lets you set the seed. This will make for easier
    //debugging in the future.
    srand(time(0));
}

template <typename K>
double** Ann<K>::newRandomMatrix(const size_t &rows, const size_t &cols, const double &max_val, const double &min_val) {
    if (max_val < min_val) {
        //TODO This should be logged.
        //TODO Format the error message to show the min and max values.
        throw logic_error("Maximum value must be greater than the minimum value.");
    }
    double **result = MatrixOps::newMatrix(rows, cols);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            const double val = (double)rand() / RAND_MAX;
            result[r][c] = min_val + val * (max_val - min_val);
        }
    }
    return result;
}

//TODO Invalid write happens in here?
template <typename K>
double** Ann<K>::feedForward(
        double **input, const size_t &input_rows, const size_t &input_cols,
        double **weights, const size_t &weights_rows, const size_t &weights_cols,
        double **bias, const size_t &bias_rows, const size_t &bias_cols,
        const size_t &output_cols) {
    assert(bias_cols == 1);
    assert(input_rows == bias_rows);
    assert(weights_rows == input_cols + 1);
    double **input_bias = MatrixOps::horizontalConcat(input, input_rows, input_cols, bias, bias_rows, bias_cols);
    //cout << "Weight: " << weights_rows << "," << weights_cols << endl;
    //cout << "Input: " << input_rows << "," << input_cols + bias_cols << endl;
    double **net = MatrixOps::multiply(input_bias, input_rows, input_cols + bias_cols, weights, weights_rows, weights_cols);
    MatrixOps::deleteMatrix(input_bias, input_rows);
    double **result = MatrixOps::newMatrix(input_rows, output_cols);
    for (size_t r = 0; r < input_rows; r++) {
        for (size_t c = 0; c < output_cols; c++) {
            result[r][c] = kernel.kernelFunc(net[r][c]);
        }
    }
    MatrixOps::deleteMatrix(net, input_rows);
    return result;
}

template <typename K>
double** Ann<K>::backProp(
        double **input, const size_t &input_rows, const size_t &input_cols,
        double **weights, const size_t &weights_rows, const size_t &weights_cols,
        double **bias, const size_t &bias_rows, const size_t &bias_cols,
        const float &rate) {
    assert(bias_cols == 1);
    assert(bias_rows == input_rows);
    assert(weights_rows == input_cols + 1);
    double **result = MatrixOps::newMatrix(weights_rows, weights_cols);
    for (size_t r = 0; r < weights_rows; r++) {
        for (size_t c = 0; c < weights_cols; c++) {
            result[r][c] = weights[r][c];
        }
    }
    return result;
}

template <typename K>
double** Ann<K>::train(
        double **training, const size_t &training_rows,
        double **validation, const size_t &validation_rows,
        double **test, const size_t &test_rows,
        const size_t &cols,
        const size_t &data_cols) {
    double **weights = newRandomMatrix(data_cols + 1, cols - data_cols, 0.5, -0.5);

    //Init Bias
    double **bias_training = MatrixOps::newMatrix(training_rows, 1);
    for (int i = 0; i < training_rows; i++) {
        bias_training[i][0] = 1;
    }
    double **bias_validation = MatrixOps::newMatrix(validation_rows, 1);
    for (int i = 0; i < validation_rows; i++) {
        bias_validation[i][0] = 1;
    }
    double **bias_test = MatrixOps::newMatrix(test_rows, 1);
    for (int i = 0; i < test_rows; i++) {
        bias_test[i][0] = 1;
    }

    const size_t class_cols = cols - data_cols;
    double **class_data = MatrixOps::split(training,
            0, training_rows,
            data_cols, class_cols);
    unsigned int *classes = MatrixOps::matrixToClass(class_data, training_rows, class_cols);

    //Loop
    int i = 0;
    while (i < 500) {
        i++;
        double **new_weights = backProp(
                training, training_rows, data_cols,
                weights, data_cols + 1, cols - data_cols,
                bias_training, training_rows, 1,
                0.1);
        MatrixOps::deleteMatrix(weights, data_cols + 1);
        weights = new_weights;
        updateError(
               training, training_rows, data_cols,
               weights, data_cols + 1, cols - data_cols,
               bias_training, training_rows, 1,
               class_data, training_rows, cols - data_cols,
               classes, training_rows);
    }

    delete[] classes;
    MatrixOps::deleteMatrix(class_data, training_rows);
    MatrixOps::deleteMatrix(bias_test, test_rows);
    MatrixOps::deleteMatrix(bias_validation, validation_rows);
    MatrixOps::deleteMatrix(bias_training, training_rows);
    return weights;
}

// This is leaking memory.
template <typename K>
void Ann<K>::updateError(
            double **input, const size_t &input_rows, const size_t &input_cols,
            double **weights, const size_t &weights_rows, const size_t &weights_cols,
            double **bias, const size_t &bias_rows, const size_t &bias_cols,
            double **target_output, const size_t &target_output_rows, const size_t &target_output_cols,
            unsigned int *target_classes, const size_t &target_classes_rows) {

    double **output = feedForward(
            input, input_rows, input_cols,
            weights, weights_rows, weights_cols,
            bias, bias_rows, bias_cols,
            target_output_cols);

    double **subtracted = MatrixOps::subtract(target_output, output, target_output_rows, target_output_cols);
    double **squared = MatrixOps::hadamardMultiply(subtracted, subtracted, target_output_rows, target_output_cols);
    training_error.push_back(MatrixOps::sum(squared, target_output_rows, target_output_cols) / (target_output_rows * target_output_cols));

    MatrixOps::deleteMatrix(subtracted, target_output_rows);
    MatrixOps::deleteMatrix(squared, target_output_rows);

    unsigned int *classes = MatrixOps::matrixToClass(output, target_output_rows, target_output_cols);
    unsigned int correct_classes = 0;
    for (size_t r = 0; r < target_classes_rows; r++) {
        if (classes[r] != target_classes[r]) {
            correct_classes++;
        }
    }
    training_classification_error.push_back((double)correct_classes / target_classes_rows);

    MatrixOps::deleteMatrix(output, target_output_rows);
    delete[] classes;
}

template<typename K>
float* Ann<K>::getTrainingError(size_t *length) {
    *length = training_error.size();
    float* result = new float[training_error.size()];
    memcpy(result, &training_error.front(), training_error.size() * sizeof(float));
    return result;
}

template<typename K>
float* Ann<K>::getTrainingClassificationError(size_t *length) {
    *length = training_classification_error.size();
    float* result = new float[training_classification_error.size()];
    memcpy(result, &training_classification_error.front(), training_classification_error.size() * sizeof(float));
    return result;
}

// List all Kernels used here.
template class Ann<LogSigmoid>;

