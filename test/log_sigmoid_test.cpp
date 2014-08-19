#include "gtest/gtest.h"
#include "log_sigmoid.hpp"

TEST(LogSigmoidTest, Sigmoid0) {
    double expectedVal = 0.5;

    LogSigmoid kernel;
    double actualVal = kernel.kernelFunc(0);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, Sigmoid1) {
    double expectedVal = 0.7310585786300048792511592418218362743651446401650565192763659079190404530702046393874745320759812453;

    LogSigmoid kernel;
    double actualVal = kernel.kernelFunc(1);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, Sigmoid2) {
    double expectedVal = 0.8807970779778824440597291413023967952063842986289682757984052500609766222883192417294737608368383572;

    LogSigmoid kernel;
    double actualVal = kernel.kernelFunc(2);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, Sigmoid3) {
    double expectedVal = 0.9525741268224332191211518482282477986138205675793908992821119912255512884972897661142163455089399610;

    LogSigmoid kernel;
    double actualVal = kernel.kernelFunc(3);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, Sigmoid4) {
    double expectedVal = 0.9820137900379084419732068620504615751275149881204673880241316087065539731588051012797374250226038446;

    LogSigmoid kernel;
    double actualVal = kernel.kernelFunc(4);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

