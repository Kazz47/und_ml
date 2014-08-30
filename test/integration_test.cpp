#include "gtest/gtest.h"
#include "matrix_ops.hpp"
#include "file_reader.hpp"
#include "ann.hpp"
#include "log_sigmoid.hpp"

TEST(IntegrationTest, Test) {
    size_t expected_cols = 7;

    char *test_file = "../../test/data/iris_test.dat";
    char *training_file = "../../test/data/iris_training.dat";
    char *validation_file = "../../test/data/iris_validation.dat";
    FileReader test_reader(test_file);
    FileReader training_reader(training_file);
    FileReader validation_reader(validation_file);
    size_t test_rows = 0;
    size_t test_cols = 0;
    size_t training_rows = 0;
    size_t training_cols = 0;
    size_t validation_rows = 0;
    size_t validation_cols = 0;
    double **test_matrix = test_reader.getMatrix(&test_rows, &test_cols);
    double **training_matrix = training_reader.getMatrix(&training_rows, &training_cols);
    double **validation_matrix = validation_reader.getMatrix(&validation_rows, &validation_cols);

    // Check that all data sets have the expected number of columns.
    ASSERT_EQ(expected_cols, test_cols);
    ASSERT_EQ(expected_cols, training_cols);
    ASSERT_EQ(expected_cols, validation_cols);

    LogSigmoid kernel;
    Ann<LogSigmoid> net(kernel);
    double **result = net.train(
            training_matrix, training_rows, training_cols,
            validation_matrix, validation_rows, validation_cols,
            test_matrix, test_rows, test_cols);

    MatrixOps::deleteMatrix(test_matrix, test_rows);
    MatrixOps::deleteMatrix(training_matrix, training_rows);
    MatrixOps::deleteMatrix(validation_matrix, validation_rows);

    for (int r = 0; r < 1; r++) {
        for (int c = 0; c < expected_cols; c++) {
            cout << result[r][c] << " ";
            ASSERT_LT(result[r][c], 0.5);
            ASSERT_GT(result[r][c], -0.5);
        }
        cout << endl;
    }

    MatrixOps::deleteMatrix(result, 1);
}
