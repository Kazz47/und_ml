#include "gtest/gtest.h"
#include "ann.hpp"
#include "matrix_ops.hpp"

TEST(AnnTest, MatrixToClass) {
    double **matrix = MatrixOps::newMatrix(5, 3);
    matrix[0][0] = 1;
    matrix[0][1] = 0;
    matrix[0][2] = 0;
    matrix[1][0] = 0;
    matrix[1][1] = 1;
    matrix[1][2] = 0;
    matrix[2][0] = 0;
    matrix[2][1] = 0;
    matrix[2][2] = 1;
    matrix[3][0] = 1;
    matrix[3][1] = 0;
    matrix[3][2] = 0;
    matrix[4][0] = 0;
    matrix[4][1] = 0;
    matrix[4][2] = 1;

    unsigned int *expectedVal = new unsigned int[5];
    expectedVal[0] = 1;
    expectedVal[1] = 2;
    expectedVal[2] = 3;
    expectedVal[3] = 1;
    expectedVal[4] = 3;

    unsigned int *actualVal = Ann::matrixToClass(matrix, 5, 3);

    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(expectedVal[i], actualVal[i]);
    }

    delete[] actualVal;
    delete[] expectedVal;
    MatrixOps::deleteMatrix(matrix, 5);
}
