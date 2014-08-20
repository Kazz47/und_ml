#include "tan_sigmoid.hpp"

double TanSigmoid::kernelFunc(const double &x) {
    return (tanh(x) + 1.0L) * 0.5L;
}

double TanSigmoid::kernelDeriv(const double &x) {
    return (1.0L - (tanh(x) * tanh(x))) * 0.5L;
}

