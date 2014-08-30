#include "gtest/gtest.h"
#include "file_reader.hpp"
#include "matrix_ops.hpp"

//TODO Fix these tests so they can be run from any directory.
TEST(FileReaderTest, ReadTestFile) {
    char *val = "../../test/data/iris_test.dat";
    FileReader reader(val);

    size_t actual_rows = 0;
    size_t actual_cols = 0;
    size_t expected_rows = 38;
    size_t expected_cols = 7;
    double **matrix = reader.getMatrix(&actual_rows, &actual_cols);

    ASSERT_EQ(expected_rows, actual_rows);
    ASSERT_EQ(expected_cols, actual_cols);

    MatrixOps::deleteMatrix(matrix, 38);
}

TEST(FileReaderTest, ReadValidationFile) {
    char *val = "../../test/data/iris_validation.dat";
    FileReader reader(val);

    size_t actual_rows = 0;
    size_t actual_cols = 0;
    size_t expected_rows = 37;
    size_t expected_cols = 7;
    double **matrix = reader.getMatrix(&actual_rows, &actual_cols);

    ASSERT_EQ(expected_rows, actual_rows);
    ASSERT_EQ(expected_cols, actual_cols);

    MatrixOps::deleteMatrix(matrix, actual_rows);
}

TEST(FileReaderTest, ReadTrainingFile) {
    char *val = "../../test/data/iris_training.dat";
    FileReader reader(val);

    size_t actual_rows = 0;
    size_t actual_cols = 0;
    size_t expected_rows = 75;
    size_t expected_cols = 7;
    double **matrix = reader.getMatrix(&actual_rows, &actual_cols);

    ASSERT_EQ(expected_rows, actual_rows);
    ASSERT_EQ(expected_cols, actual_cols);

    MatrixOps::deleteMatrix(matrix, actual_rows);
}
