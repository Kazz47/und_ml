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

    /**
     * Method to pass inputs through a ANN and get classification results.
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

private:
    K kernel;
};

#endif //ANN_H

