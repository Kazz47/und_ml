#include "gtest/gtest.h"
#include "matrix_ops.hpp"

TEST(AnnTest, SuccessfulBuild) {
    SUCCEED();
}

TEST(AnnTest, SingleItemTranspose) {
    // Setup
    double **matrix = new double*[1];
    for (int i = 0; i < 1; i++) {
        matrix[i] = new double[1];
    }
    matrix[0][0] = 1;

    double **expectedVal = new double*[1];
    for (int i = 0; i < 1; i++) {
        expectedVal[i] = new double[1];
    }
    expectedVal[0][0] = 1;

    // Test
    double** actualVal = MatrixOps::transpose(matrix, 1, 1);

    ASSERT_EQ(expectedVal[0][0], actualVal[0][0]);

    // Tear Down
    for (int i = 0; i < 1; i++) {
        delete[] actualVal[i];
    }
    delete[] actualVal;

    for (int i = 0; i < 1; i++) {
        delete[] expectedVal[i];
    }
    delete[] expectedVal;

    for (int i = 0; i < 1; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

TEST(AnnTest, SquareMatrixTranspose) {
    // Setup
    double **matrix = new double*[3];
    for (int i = 0; i < 3; i++) {
        matrix[i] = new double[3];
    }
    matrix[0][0] = 1;
    matrix[0][1] = 2;
    matrix[0][2] = 3;
    matrix[1][0] = 4;
    matrix[1][1] = 5;
    matrix[1][2] = 6;
    matrix[2][0] = 7;
    matrix[2][1] = 8;
    matrix[2][2] = 9;

    double **expectedVal = new double*[3];
    for (int i = 0; i < 3; i++) {
        expectedVal[i] = new double[3];
    }
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
    for (int i = 0; i < 3; i++) {
        delete[] actualVal[i];
    }
    delete[] actualVal;

    for (int i = 0; i < 3; i++) {
        delete[] expectedVal[i];
    }
    delete[] expectedVal;

    for (int i = 0; i < 3; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

TEST(AnnTest, MatrixTranspose) {
    // Setup
    double **matrix = new double*[2];
    for (int i = 0; i < 2; i++) {
        matrix[i] = new double[3];
    }
    matrix[0][0] = 1;
    matrix[0][1] = 2;
    matrix[0][2] = 3;
    matrix[1][0] = 4;
    matrix[1][1] = 5;
    matrix[1][2] = 6;

    double **expectedVal = new double*[3];
    for (int i = 0; i < 3; i++) {
        expectedVal[i] = new double[2];
    }
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
    for (int i = 0; i < 3; i++) {
        delete[] actualVal[i];
    }
    delete[] actualVal;

    for (int i = 0; i < 3; i++) {
        delete[] expectedVal[i];
    }
    delete[] expectedVal;

    for (int i = 0; i < 2; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

