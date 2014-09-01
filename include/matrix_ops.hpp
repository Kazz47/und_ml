#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <cstddef>
#include <stdexcept>

using namespace std;

/**
 * Class that contains basic matrix operations.
 */
class MatrixOps {
public:

    /**
     * Method that creates a new matrix.
     * This method allocates memory for a matrix of the specified size.
     * Call {@link deleteMatrix} to easily delete the matrix.
     *
     * @param rows Number of rows in the new matrix.
     * @param cols Number of columns in the new matrix.
     * @return New matrix that with the specified number of rows and columns.
     */
    static double** newMatrix(const size_t &rows, const size_t &cols);

    /**
     * Method that deletes a matrix.
     * This method deallocates the memory of a matrix.
     *
     * @param matrix Matrix to delete.
     * @param rows Number of rows in the matrix to delete.
     */
    static void deleteMatrix(double **&matrix, const size_t &rows);

    /**
     * Method that returns the transpose of a matrix.
     * This method allocates memory for the new matrix.
     *
     * @param matrix Input matrix.
     * @param rows Number of rows in the input matrix.
     * @param cols Number of columns in the input matrix.
     * @return New matrix that is the transpose of the input matrix.
     */
    static double** transpose(double **matrix, const size_t &rows, const size_t &cols);

    /**
     * Method that returns the addition of two matrices.
     * This is a component-wise addition of each item in matrix_one with the
     * corresponding item in matrix_two.
     * This method allocates memory for the new matrix and will not modify the
     * input matrices.
     *
     * @param matrix_one The first input matrix.
     * @param matrix_two The second input matrix.
     * @param rows Number of rows in the input matrices.
     * @param cols Number of columns in the input matrices.
     * @return New matrix that is the addition of the two input matrices.
     */
    static double** add(double **matrix_one, double **matrix_two, const size_t &rows, const size_t &cols);

    /**
     * Method that returns the subtraction of two matrices.
     * This is a component-wise subtraction of each item in matrix_two from the
     * corresponding item in matrix_one.
     * This method allocates memory for the new matrix and will not modify the
     * input matrices.
     *
     * @param matrix_one The first input matrix.
     * @param matrix_two The second input matrix.
     * @param rows Number of rows in the input matrices.
     * @param cols Number of columns in the input matrices.
     * @return New matrix that is the subtraction of the two input matrices.
     */
    static double** subtract(double **matrix_one, double **matrix_two, const size_t &rows, const size_t &cols);

    /**
     * Method that returns a matrix multiplied by a scalar.
     * This method allocates memory for the new matrix and will not modify the
     * input matrix.
     *
     * @param matrix Matrix to multiply.
     * @param scalar Scalar to multiply the matrix.
     * @param rows Number of rows in the input matrix.
     * @param cols Number of columns in the input matrix.
     * @return New matrix that is the input matrix multiplied by the provided
     * scalar.
     */
    static double** scalarMultiply(double **matrix, const double &scalar, const size_t &rows, const size_t &cols);

    /**
     * Method that returns a matrix that is the product of two multiplied
     * matrices.
     * This method allocates memory for the new matrix and will not modify the
     * input matrix.
     *
     * @param matrix_one The first input matrix.
     * @param rows_one Number of rows in matrix_one.
     * @param cols_one Number of columns in matrix_two.
     * @param matrix_two The second input matrix.
     * @param rows_two Number of rows in matrix_one.
     * @param cols_two Number of columns in matrix_two.
     * @return New matrix that is the product of the two input matrices.
     * @throw logic_error
     */
    static double** multiply(
            double **matrix_one, const size_t &rows_one, const size_t &cols_one,
            double **matrix_two, const size_t &rows_two, const size_t &cols_two);

    /**
     * Method that returns a matrix that is the hadamard product of two
     * matrices.
     * This method allocates memory for the new matrix and will not modify the
     * input matrices.
     *
     * @param matrix_one The first input matrix.
     * @param matrix_two The second input matrix.
     * @param rows Number of rows in the matrices.
     * @param cols Number of columns in the matrices.
     * @return New matrix that is hadamard product of the two input matrices.
     */
    static double** hadamardMultiply(double **matrix_one, double **matrix_two, const size_t &rows, const size_t &cols);

    /**
     * Method that returns a matrix that is the kronecker product of two matrices.
     * This method allocates memory for the new matrix and will not modify the
     * input matrices.
     *
     * @param matrix_one The first input matrix.
     * @param rows_one Number of rows in matrix_one.
     * @param cols_one Number of columns in matrix_two.
     * @param matrix_two The second input matrix.
     * @param rows_two Number of rows in matrix_one.
     * @param cols_two Number of columns in matrix_two.
     * @return New matrix that is the kronecker product of the two input matrices.
     */
    static double** kroneckerMultiply(
            double **matrix_one, const size_t &rows_one, const size_t &cols_one,
            double **matrix_two, const size_t &rows_two, const size_t &cols_two);

    /**
     * Method that returns a matrix that is the horizontal concatenation of two
     * input matrices.
     * Thie method allocates mempory for the new matrix and will not modify the
     * input matrices.
     *
     * @param matrix_one The first input matrix.
     * @param rows_one Number of rows in matrix_one.
     * @param cols_one Number of columns in matrix_two.
     * @param matrix_two The second input matrix.
     * @param rows_two Number of rows in matrix_one.
     * @param cols_two Number of columns in matrix_two.
     * @return New matrix that is the horizontal concatenation of the two input matrices.
     * @throw logic_error
     */
    static double** horizontalConcat(
            double **matrix_one, const size_t &rows_one, const size_t &cols_one,
            double **matrix_two, const size_t &rows_two, const size_t &cols_two);

    /**
     * Method that returns a matrix that is the vertical concatenation of two
     * input matrices.
     * Thie method allocates mempory for the new matrix and will not modify the
     * input matrices.
     *
     * @param matrix_one The first input matrix.
     * @param rows_one Number of rows in matrix_one.
     * @param cols_one Number of columns in matrix_two.
     * @param matrix_two The second input matrix.
     * @param rows_two Number of rows in matrix_one.
     * @param cols_two Number of columns in matrix_two.
     * @return New matrix that is the vertical concatenation of the two input matrices.
     * @throw logic_error
     */
    static double** verticalConcat(
            double **matrix_one, const size_t &rows_one, const size_t &cols_one,
            double **matrix_two, const size_t &rows_two, const size_t &cols_two);

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

#endif //MATRIX_OPS_H

