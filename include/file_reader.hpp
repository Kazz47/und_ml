#ifndef FILE_READER_H
#define FILE_READER_H

#include <cstddef>
#include <fstream>
#include <string>

#include "matrix_reader.hpp"

using namespace std;

/**
 * File reader object for loading a matrix from a file.
 */
class FileReader : public MatrixReader {
public:

    /**
     * File reader constructor.
     * Opens the specified file name for reading.
     */
    FileReader(char *file_name);

    /**
     * File reader deconstructor.
     * Closes the file associated with this object.
     */
    ~FileReader();

    /**
     * Get a matrix from the file.
     * Call {@link MatrixOps::deleteMatrix} to easily delete the matrix.
     *
     * @param rows Pointer to store the number of rows in the matrix.
     * @param cols Pointer to store the number of columns in the matrix.
     * @return Matrix and the number of rows and columns.
     */
    double** getMatrix(size_t *rows, size_t *cols);

private:
    ifstream infile;
};

#endif //FILE_READER_H
