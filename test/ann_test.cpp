#include "gtest/gtest.h"
#include "matrix_ops.hpp"
#include "ann.hpp"
#include "log_sigmoid.hpp"

TEST(AnnTest, WeightInitilzation) {
    LogSigmoid kernel;
    Ann<LogSigmoid> net(kernel);
    double max_val = 50;
    double min_val = 10;
    double **matrix = net.newRandomMatrix(2, 2, max_val, min_val);

    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 2; c++) {
            ASSERT_LT(matrix[r][c], max_val);
            ASSERT_GT(matrix[r][c], min_val);
        }
    }

    MatrixOps::deleteMatrix(matrix, 2);
}

TEST(AnnTest, DISABLED_FeedForwardLogSigmoidNoBias) {
    double **input = MatrixOps::newMatrix(2, 2);
    input[0][0] = 1;
    input[0][1] = 2;
    input[1][0] = 3;
    input[1][1] = 4;

    double **weights = MatrixOps::newMatrix(2, 2);
    weights[0][0] = 0;
    weights[0][1] = 0;
    weights[1][0] = 0;
    weights[1][1] = 0;

    double **bias = MatrixOps::newMatrix(2, 1);
    bias[0][0] = 0;
    bias[0][1] = 0;

    double **expectedVal = MatrixOps::newMatrix(2, 2);
    expectedVal[0][0] = 0.7310585786300048792511592418218362743651446401650565192763659079190404530702046393874745320759812453;
    expectedVal[0][1] = 0.8807970779778824440597291413023967952063842986289682757984052500609766222883192417294737608368383572;
    expectedVal[1][0] = 0.9525741268224332191211518482282477986138205675793908992821119912255512884972897661142163455089399610;
    expectedVal[1][1] = 0.9820137900379084419732068620504615751275149881204673880241316087065539731588051012797374250226038446;

    LogSigmoid kernel;
    Ann<LogSigmoid> net(kernel);
    double **actualVal = net.feedForward(
            input, 2, 2,
            weights, 2, 2,
            bias, 2, 1);
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            ASSERT_DOUBLE_EQ(expectedVal[r][c], actualVal[r][c]);
        }
    }

    MatrixOps::deleteMatrix(actualVal, 2);
    MatrixOps::deleteMatrix(expectedVal, 2);
    MatrixOps::deleteMatrix(bias, 2);
    MatrixOps::deleteMatrix(weights, 2);
    MatrixOps::deleteMatrix(input, 2);
}

//TODO This needs to be updated to be update to actually test backprop.
TEST(AnnTest, BackProp) {
    double **input = MatrixOps::newMatrix(2, 2);
    input[0][0] = 1;
    input[0][1] = 2;
    input[1][0] = 3;
    input[1][1] = 4;

    double **weights = MatrixOps::newMatrix(2, 2);
    weights[0][0] = 0;
    weights[0][1] = 0;
    weights[1][0] = 0;
    weights[1][1] = 0;

    double **bias = MatrixOps::newMatrix(2, 1);
    bias[0][0] = 0;
    bias[0][1] = 0;

    double **expectedVal = MatrixOps::newMatrix(2, 2);
    expectedVal[0][0] = 0;
    expectedVal[0][1] = 0;
    expectedVal[1][0] = 0;
    expectedVal[1][1] = 0;

    LogSigmoid kernel;
    Ann<LogSigmoid> net(kernel);
    double **actualVal = net.backProp(
            input, 2, 2,
            weights, 2, 2,
            bias, 2, 1,
            1);
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            ASSERT_DOUBLE_EQ(expectedVal[r][c], actualVal[r][c]);
        }
    }

    MatrixOps::deleteMatrix(actualVal, 2);
    MatrixOps::deleteMatrix(expectedVal, 2);
    MatrixOps::deleteMatrix(bias, 2);
    MatrixOps::deleteMatrix(weights, 2);
    MatrixOps::deleteMatrix(input, 2);
}

//TODO Update this test, currently it just checks that the train method returns
//randomly genereated weights.
TEST(AnnTest, Train) {
    double **training = MatrixOps::newMatrix(2, 2);
    training[0][0] = 1;
    training[0][1] = 2;
    training[1][0] = 3;
    training[1][1] = 4;

    double **validation = MatrixOps::newMatrix(2, 2);
    validation[0][0] = 0;
    validation[0][1] = 0;
    validation[1][0] = 0;
    validation[1][1] = 0;

    double **test = MatrixOps::newMatrix(2, 1);
    test[0][0] = 0;
    test[1][0] = 0;

    LogSigmoid kernel;
    Ann<LogSigmoid> net(kernel);
    double **actualVal = net.train(
            training, 2, 2,
            validation, 2, 2,
            test, 2, 2);
    for (int r = 0; r < 1; r++) {
        for (int c = 0; c < 2; c++) {
            ASSERT_LT(actualVal[r][c], 0.5);
            ASSERT_GT(actualVal[r][c], -0.5);
        }
    }

    MatrixOps::deleteMatrix(actualVal, 1);
    MatrixOps::deleteMatrix(test, 2);
    MatrixOps::deleteMatrix(validation, 2);
    MatrixOps::deleteMatrix(training, 2);
}

