#ifndef ANN_H
#define ANN_H

#include <cstddef>

using namespace std;

/**
 * Class for training and using an Artifical Neural Network (ANN).
 */
template <typename K>
class Ann {
public:

    Ann(const K kernel);
    //~Ann();

    /**
     * Method that creates a new matrix filled with random values.
     * This method allocates memory for a matrix of the specified size.
     * Call {@link MatrixOps::deleteMatrix} to easily delete the matrix.
     *
     * @param rows Number of rows in the new matrix.
     * @param cols Number of columns in the new matrix.
     * @param max_val Largest possible random value.
     * @param min_val Smallest possible random value.
     * @return New matrix that with the specified number of rows and columns
     * initialized to values betwen the specified min and max.
     * @throw logic_error
     */
    double** newRandomMatrix(const size_t &rows, const size_t &cols, const double &max_val, const double &min_val);

    //TODO Update comment. How big of a matrix should this be?
    /**
     * Method to pass inputs through the ANN and get classification results.
     * This method allocates memory for a matrix of the size ?.
     * Call {@link MatrixOps::deleteMatrix} to easily delete the matrix.
     *
     * @param input Matrix of input values for the ANN.
     * @param input_rows Number of rows in the input matrix.
     * @param input_cols Number of columns in the input matrix.
     * @param weights Matrix of node weights for the ANN.
     * @param weights_rows Number of rows in the weights matrix.
     * @param weights_cols Number of columns in the weights matrix.
     * @param bias Matrix of node bias for the ANN.
     * @param bias_rows Number of rows in the bias matrix.
     * @param bias_cols Number of columns in the bias matrix.
     * @return The resulting classification matrix.
     */
    double** feedForward(
            double **input, const size_t &input_rows, const size_t &input_cols,
            double **weights, const size_t &weights_rows, const size_t &weights_cols,
            double **bias, const size_t &bias_rows, const size_t &bias_cols);

    /**
     * Method to backpropagate through the ANN and get an updated weight matrix.
     * This method allocates memory for a matrix of the size of 'weights_rows'
     * by 'weights_cols'.
     * Call {@link MatrixOps::deleteMatrix} to easily delete the matrix.
     *
     * @param input Matrix of input values for the ANN.
     * @param input_rows Number of rows in the input matrix.
     * @param input_cols Number of columns in the input matrix.
     * @param weights Matrix of node weights for the ANN.
     * @param weights_rows Number of rows in the weights matrix.
     * @param weights_cols Number of columns in the weights matrix.
     * @param bias Matrix of node bias for the ANN.
     * @param bias_rows Number of rows in the bias matrix.
     * @param bias_cols Number of columns in the bias matrix.
     * @param rate Learning rate for backprop.
     * @return The resulting updated weigth matrix.
     */
    double** backProp(
            double **input, const size_t &input_rows, const size_t &input_cols,
            double **weights, const size_t &weights_rows, const size_t &weights_cols,
            double **bias, const size_t &bias_rows, const size_t &bias_cols,
            const float &rate);

    //TODO Update comment. How big of a matrix should thiS be?
    /**
     * Method to traing an ANN and return the resulting weight matrix.
     * This method allocates memory for a matrix of size ?.
     * Call {@link MatrixOps::deleteMatrix} to easily delete the matrix.
     *
     * @param input Matrix of input values for the ANN.
     * @param input_rows Number of rows in the input matrix.
     * @param input_cols Number of columns in the input matrix.
     * @param weights Matrix of node weights for the ANN.
     * @param weights_rows Number of rows in the weights matrix.
     * @param weights_cols Number of columns in the weights matrix.
     * @param bias Matrix of node bias for the ANN.
     * @param bias_rows Number of rows in the bias matrix.
     * @param bias_cols Number of columns in the bias matrix.
     * @return The resulting weight matrix.
     */
    double** train(
            double **training, const size_t &training_rows, const size_t &training_cols,
            double **validation, const size_t &validation_rows, const size_t &validation_cols,
            double **test, const size_t &test_rows, const size_t &test_cols);

private:
    K kernel;
};

#endif //ANN_H

