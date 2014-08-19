#ifndef ANN_H
#define ANN_H

#include <cstddef>

using namespace std;

/**
 * Class for training and using an Artifical Neural Network (ANN).
 */
class Ann {
public:
    /**
     * Method to pass inputs through a ANN and get classification results.
     *
     * @param input Matrix of input values for the ANN.
     * @param weights Matrix of weights for the ANN nodes.
     * @param bias Matrix of bias for the ANN nodes.
     * @return The resulting classification matrix.
     */
    static double** feedForward(double **input, double **weights, double **bias);

    /**
     * Method to parse a matrix and return a vector of classes.
     * Classes start with a value of 1 and are incremented for each new class.
     *
     * @param matrix Matrix to vectorize into classes.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @return The vector of classes for each matrix row.
     */
    static unsigned int* matrixToClass(double **matrix, const size_t &rows, const size_t &cols);

private:

    /*
     * Method to check each row in a matrix for the values in a provided array.
     *
     * @param row Array of values to find.
     * @param matrix Matrix to search.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @return The row index of the row if found, otherwise -1.
     */
    static long rowInMatrix(double *row, double **matrix, const size_t &rows, const size_t &cols);
};

#endif //ANN_H

