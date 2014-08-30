#ifndef MATRIX_READER_H
#define MATRIX_READER_H

#include <cstddef>

/**
 * Generic matrix reader for loading a matrix.
 */
class MatrixReader {
public:

    /**
     * Get a matrix from a data source.
     * Call {@link MatrixOps::deleteMatrix} to easily delete the matrix.
     *
     * @param rows Pointer to store the number of rows in the matrix.
     * @param cols Pointer to store the number of columns in the matrix.
     * @return Matrix and the number of rows and columns.
     */
    virtual double** getMatrix(size_t *rows, size_t *cols) = 0;
};

#endif //MATRIX_READER_H
