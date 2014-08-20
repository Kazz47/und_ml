#include "gtest/gtest.h"
#include "tan_sigmoid.hpp"

TEST(TanSigmoidTest, Sigmoid0) {
    double expectedVal = 0.5;

    TanSigmoid kernel;
    double actualVal = kernel.kernelFunc(0);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(TanSigmoidTest, Sigmoid1) {
    double expectedVal = 0.8807970779778824440597291413023967952063842986289682757984052500609766222883192417294737608368383572;

    TanSigmoid kernel;
    double actualVal = kernel.kernelFunc(1);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(TanSigmoidTest, Sigmoid2) {
    double expectedVal = 0.9820137900379084419732068620504615751275149881204673880241316087065539731588051012797374250226038446;

    TanSigmoid kernel;
    double actualVal = kernel.kernelFunc(2);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(TanSigmoidTest, Sigmoid3) {
    double expectedVal = 0.9975273768433652256659400926277442375489069273501412459119394075653323513912795883596979800111558790;

    TanSigmoid kernel;
    double actualVal = kernel.kernelFunc(3);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(TanSigmoidTest, Sigmoid4) {
    double expectedVal = 0.9996646498695335218961216721708624810026699264472048240039034482283394430591600901146977137361448686;

    TanSigmoid kernel;
    double actualVal = kernel.kernelFunc(4);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(TanSigmoidTest, SigmoidDeriv0) {
    double expectedVal = 0.5;

    TanSigmoid kernel;
    double actualVal = kernel.kernelDeriv(0);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(TanSigmoidTest, SigmoidDeriv1) {
    double expectedVal = 0.2099871708070130346972483695208507224585933641153854773566557201222949497620241528078470044311593630;

    TanSigmoid kernel;
    double actualVal = kernel.kernelDeriv(1);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(TanSigmoidTest, SigmoidDeriv2) {
    double expectedVal = 0.03532541242658223284312382793052739615606362991882798056243190491343103030472792095272482867191154543;

    TanSigmoid kernel;
    double actualVal = kernel.kernelDeriv(2);

    ASSERT_DOUBLE_EQ(expectedVal, actualVal);
}

TEST(TanSigmoidTest, SigmoidDeriv3) {
    double expectedVal = 0.004933018582720095636578084841761743663660804559399863602124745023211662148054750855857728873900145776;

    TanSigmoid kernel;
    double actualVal = kernel.kernelDeriv(3);

    ASSERT_NEAR(expectedVal, actualVal, 0.0000000000000001L);
}

TEST(TanSigmoidTest, SigmoidDeriv4) {
    double expectedVal = 0.0006704753415129484399851094516655066155078830348121154630238921104701595170235635223347777256572355192;


    TanSigmoid kernel;
    double actualVal = kernel.kernelDeriv(4);

    ASSERT_NEAR(expectedVal, actualVal, 0.0000000000000001L);
}

