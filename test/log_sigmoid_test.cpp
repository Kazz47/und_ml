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

TEST(LogSigmoidTest, SigmoidDeriv0) {
    double expectedVal = 0.25;

    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(0);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, SigmoidDeriv1) {
    double expectedVal = 0.1966119332414818525374247335859090256222672854273135775763268324868302873926450596563475285482006501;

    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(1);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, SigmoidDeriv2) {
    double expectedVal = 0.1049935854035065173486241847604253612292966820576927386783278600611474748810120764039235022155796815;

    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(2);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, SigmoidDeriv3) {
    double expectedVal = 0.04517665973091213264936002843565163799485995239748969246454782778480922682026316173926092989723483536;

    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(3);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, SigmoidDeriv4) {
    double expectedVal = 0.01766270621329111642156191396526369807803181495941399028121595245671551515236396047636241433595577272;


    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(4);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

