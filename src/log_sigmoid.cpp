#include "log_sigmoid.hpp"

double LogSigmoid::kernelFunc(const double &x) {
    return 1 / (1 + exp(-x));
}

double LogSigmoid::kernelDeriv(const double &x) {
    const double eToTheX = exp(x);
    return eToTheX / ((eToTheX + 1) * (eToTheX + 1));
}

