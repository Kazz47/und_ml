#ifndef LOG_SIGMOID_H
#define LOG_SIGMOID_H

#include <cmath>
#include "kernel.hpp"

/**
 * Log sigmoid kernel.
 */
class LogSigmoid : public Kernel {
public:
    /*
     * Function that computes the log-sigmoid function value at the given 'x'.
     *
     * @param x Input value.
     * @return The value of the log-sigmoid function at the given x value.
     */
    double kernelFunc(const double &x);

    /*
     * Function that computes the log-sigmoid derivative value at the given 'x'.
     *
     * @param x Input value.
     * @return The derivative of the log-sigmoid function at the given x value.
     */
    double kernelDeriv(const double &x);
};

#endif //LOG_SIGMOID_H

