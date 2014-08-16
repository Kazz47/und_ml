#include "gtest/gtest.h"
#include "matrix_ops.hpp"

TEST(TestTest, SuccessfulBuild) {
    SUCCEED();
}

TEST(MatrixTest, MakeNewMatrix_NotNull) {
    double **actualVal = MatrixOps::newMatrix(1, 1);
    double **expectedVal = NULL;

    ASSERT_NE(expectedVal, actualVal);

    for (int i = 0; i < 1; i++) {
        delete[] actualVal[i];
    }
    delete[] actualVal;
    actualVal = NULL;
}

TEST(MatrixTest, DeleteMatrix_IsNull) {
    double **actualVal = new double*[1];
    for (int i = 0; i < 1; i++) {
        actualVal[i] = new double[1];
    }
    actualVal[0][0] = 1;

    double **expectedVal = NULL;

    MatrixOps::deleteMatrix(actualVal, 1);

    ASSERT_EQ(expectedVal, actualVal);
}

TEST(MatrixOpsTest, SingleItemTranspose) {
    // Setup
    double **matrix = MatrixOps::newMatrix(1, 1);
    matrix[0][0] = 1;

    double **expectedVal = MatrixOps::newMatrix(1, 1);
    expectedVal[0][0] = 1;

    // Test
    double** actualVal = MatrixOps::transpose(matrix, 1, 1);

    ASSERT_EQ(expectedVal[0][0], actualVal[0][0]);

    // Tear Down
    MatrixOps::deleteMatrix(actualVal, 1);
    MatrixOps::deleteMatrix(expectedVal, 1);
    MatrixOps::deleteMatrix(matrix, 1);
}

TEST(MatrixOpsTest, SquareMatrixTranspose) {
    // Setup
    double **matrix = MatrixOps::newMatrix(3, 3);
    matrix[0][0] = 1;
    matrix[0][1] = 2;
    matrix[0][2] = 3;
    matrix[1][0] = 4;
    matrix[1][1] = 5;
    matrix[1][2] = 6;
    matrix[2][0] = 7;
    matrix[2][1] = 8;
    matrix[2][2] = 9;

    double **expectedVal = MatrixOps::newMatrix(3, 3);
    expectedVal[0][0] = 1;
    expectedVal[1][0] = 2;
    expectedVal[2][0] = 3;
    expectedVal[0][1] = 4;
    expectedVal[1][1] = 5;
    expectedVal[2][1] = 6;
    expectedVal[0][2] = 7;
    expectedVal[1][2] = 8;
    expectedVal[2][2] = 9;

    // Test
    double** actualVal = MatrixOps::transpose(matrix, 3, 3);

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            ASSERT_EQ(expectedVal[r][c], actualVal[r][c]);
        }
    }

    // Tear Down
    MatrixOps::deleteMatrix(actualVal, 3);
    MatrixOps::deleteMatrix(expectedVal, 3);
    MatrixOps::deleteMatrix(matrix, 3);
}

TEST(MatrixOpsTest, MatrixTranspose) {
    // Setup
    double **matrix = MatrixOps::newMatrix(2, 3);
    matrix[0][0] = 1;
    matrix[0][1] = 2;
    matrix[0][2] = 3;
    matrix[1][0] = 4;
    matrix[1][1] = 5;
    matrix[1][2] = 6;

    double **expectedVal = MatrixOps::newMatrix(3, 2);
    expectedVal[0][0] = 1;
    expectedVal[1][0] = 2;
    expectedVal[2][0] = 3;
    expectedVal[0][1] = 4;
    expectedVal[1][1] = 5;
    expectedVal[2][1] = 6;

    // Test
    double** actualVal = MatrixOps::transpose(matrix, 2, 3);

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 2; c++) {
            ASSERT_EQ(expectedVal[r][c], actualVal[r][c]);
        }
    }

    // Tear Down
    MatrixOps::deleteMatrix(actualVal, 3);
    MatrixOps::deleteMatrix(expectedVal, 3);
    MatrixOps::deleteMatrix(matrix, 2);
}

