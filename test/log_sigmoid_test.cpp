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

TEST(LogSigmoidTest, DISABLED_SigmoidDeriv0) {
    double expectedVal = 0.5;

    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(0);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, DISABLED_SigmoidDeriv1) {
    double expectedVal = 0.2099871708070130346972483695208507224585933641153854773566557201222949497620241528078470044311593630;

    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(1);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, DISABLED_SigmoidDeriv2) {
    double expectedVal = 0.03532541242658223284312382793052739615606362991882798056243190491343103030472792095272482867191154543;

    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(2);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(LogSigmoidTest, DISABLED_SigmoidDeriv3) {
    double expectedVal = 0.004933018582720095636578084841761743663660804559399863602124745023211662148054750855857728873900145776;

    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(3);

    ASSERT_NEAR(expectedVal, actualVal, 0.0000000000000001L);
}

TEST(LogSigmoidTest, DISABLED_SigmoidDeriv4) {
    double expectedVal = 0.0006704753415129484399851094516655066155078830348121154630238921104701595170235635223347777256572355192;


    LogSigmoid kernel;
    double actualVal = kernel.kernelDeriv(4);

    ASSERT_NEAR(expectedVal, actualVal, 0.0000000000000001L);
}

