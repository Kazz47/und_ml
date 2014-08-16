#include "gtest/gtest.h"
#include "matrix_ops.hpp"

TEST(TestTest, SuccessfulBuild) {
    SUCCEED();
}

TEST(MatrixTest, MakeNewMatrixNotNull) {
    double **actualVal = MatrixOps::newMatrix(1, 1);
    double **expectedVal = NULL;

    ASSERT_NE(expectedVal, actualVal);

    for (int i = 0; i < 1; i++) {
        delete[] actualVal[i];
    }
    delete[] actualVal;
    actualVal = NULL;
}

TEST(MatrixTest, DeleteMatrixIsNull) {
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

TEST(MatrixOpsTest, MatrixAdd) {
    // Setup
    double **matrix_one = MatrixOps::newMatrix(3, 3);
    matrix_one[0][0] = 1;
    matrix_one[0][1] = 2;
    matrix_one[0][2] = 3;
    matrix_one[1][0] = 4;
    matrix_one[1][1] = 5;
    matrix_one[1][2] = 6;
    matrix_one[2][0] = 7;
    matrix_one[2][1] = 8;
    matrix_one[2][2] = 9;

    double **matrix_two = MatrixOps::newMatrix(3, 3);
    matrix_two[0][0] = 9;
    matrix_two[0][1] = 8;
    matrix_two[0][2] = 7;
    matrix_two[1][0] = 6;
    matrix_two[1][1] = 5;
    matrix_two[1][2] = 4;
    matrix_two[2][0] = 3;
    matrix_two[2][1] = 2;
    matrix_two[2][2] = 1;

    double **expectedVal = MatrixOps::newMatrix(3, 3);
    expectedVal[0][0] = 10;
    expectedVal[0][1] = 10;
    expectedVal[0][2] = 10;
    expectedVal[1][0] = 10;
    expectedVal[1][1] = 10;
    expectedVal[1][2] = 10;
    expectedVal[2][0] = 10;
    expectedVal[2][1] = 10;
    expectedVal[2][2] = 10;

    // Test
    double** actualVal = MatrixOps::add(matrix_one, matrix_two, 3, 3);

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            ASSERT_EQ(expectedVal[r][c], actualVal[r][c]);
        }
    }

    // Tear Down
    MatrixOps::deleteMatrix(actualVal, 3);
    MatrixOps::deleteMatrix(expectedVal, 3);
    MatrixOps::deleteMatrix(matrix_two, 3);
    MatrixOps::deleteMatrix(matrix_one, 3);
}

TEST(MatrixOpsTest, ScalarMatrixMultiply) {
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

    double scalar = 2.0;

    double **expectedVal = MatrixOps::newMatrix(3, 3);
    expectedVal[0][0] = 2;
    expectedVal[0][1] = 4;
    expectedVal[0][2] = 6;
    expectedVal[1][0] = 8;
    expectedVal[1][1] = 10;
    expectedVal[1][2] = 12;
    expectedVal[2][0] = 14;
    expectedVal[2][1] = 16;
    expectedVal[2][2] = 18;

    // Test
    double** actualVal = MatrixOps::multiply(matrix, scalar, 3, 3);

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

TEST(MatrixOpsTest, MatrixMultiply) {
    // Setup
    double **matrix_one = MatrixOps::newMatrix(3, 2);
    matrix_one[0][0] = 1;
    matrix_one[0][1] = 4;
    matrix_one[1][0] = 2;
    matrix_one[1][1] = 5;
    matrix_one[2][0] = 3;
    matrix_one[2][1] = 6;

    double **matrix_two = MatrixOps::newMatrix(2, 3);
    matrix_two[0][0] = 7;
    matrix_two[0][1] = 8;
    matrix_two[0][2] = 9;
    matrix_two[1][0] = 10;
    matrix_two[1][1] = 11;
    matrix_two[1][2] = 12;

    double **expectedVal = MatrixOps::newMatrix(3, 3);
    expectedVal[0][0] = 47;
    expectedVal[0][1] = 52;
    expectedVal[0][2] = 57;
    expectedVal[1][0] = 64;
    expectedVal[1][1] = 71;
    expectedVal[1][2] = 78;
    expectedVal[2][0] = 81;
    expectedVal[2][1] = 90;
    expectedVal[2][2] = 99;

    // Test
    double** actualVal = MatrixOps::multiply(matrix_one, 3, 2, matrix_two, 2, 3);

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            ASSERT_EQ(expectedVal[r][c], actualVal[r][c]);
        }
    }

    // Tear Down
    MatrixOps::deleteMatrix(actualVal, 3);
    MatrixOps::deleteMatrix(expectedVal, 3);
    MatrixOps::deleteMatrix(matrix_two, 2);
    MatrixOps::deleteMatrix(matrix_one, 3);
}

TEST(MatrixOpsTest, MatrixMultiplyBadSizes) {
    // Setup
    double **matrix_one = MatrixOps::newMatrix(3, 1);
    matrix_one[0][0] = 1;
    matrix_one[1][0] = 2;
    matrix_one[2][0] = 3;

    double **matrix_two = MatrixOps::newMatrix(2, 3);
    matrix_two[0][0] = 7;
    matrix_two[0][1] = 8;
    matrix_two[0][2] = 9;
    matrix_two[1][0] = 10;
    matrix_two[1][1] = 11;
    matrix_two[1][2] = 12;

    double **expectedVal = MatrixOps::newMatrix(3, 3);
    expectedVal[0][0] = 47;
    expectedVal[0][1] = 52;
    expectedVal[0][2] = 57;
    expectedVal[1][0] = 64;
    expectedVal[1][1] = 71;
    expectedVal[1][2] = 78;
    expectedVal[2][0] = 81;
    expectedVal[2][1] = 90;
    expectedVal[2][2] = 99;

    // Test
    try {
        double** actualVal = MatrixOps::multiply(matrix_one, 3, 1, matrix_two, 2, 3);

        MatrixOps::deleteMatrix(actualVal, 3);
        FAIL();
    } catch(logic_error e) {
        SUCCEED();
    }

    // Tear Down
    MatrixOps::deleteMatrix(expectedVal, 3);
    MatrixOps::deleteMatrix(matrix_two, 2);
    MatrixOps::deleteMatrix(matrix_one, 3);
}
