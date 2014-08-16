#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <cstddef>

using namespace std;

/**
 * Class that contains basic matrix operations.
 */
class MatrixOps {
public:
    /**
     * Method that returns the transpose of a matrix.
     * This method allocates memory for the new matrix.
     *
     * @param matrix Input matrix.
     * @param rows Number of rows in the input matrix.
     * @param cols Number of columns in the input matrix.
     * @return New matrix that is the transpose of the input matrix.
     */
    static double** transpose(double **matrix, const size_t rows, const size_t cols);
};

#endif //MATRIX_OPS_H
