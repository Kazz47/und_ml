#include "log_sigmoid.hpp"

#include <cmath>

double LogSigmoid::kernelFunc(const double &x) {
    return 1.0L / (1.0L + exp(-x));
}

double LogSigmoid::kernelDeriv(const double &x) {
    double eToTheX = exp(x);
    return eToTheX / ((eToTheX + 1.0L) * (eToTheX + 1.0L));
}

