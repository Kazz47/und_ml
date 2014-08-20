#ifndef TAN_SIGMOID_H
#define TAN_SIGMOID_H

#include <cmath>
#include "kernel.hpp"

/**
 * Tan sigmoid kernel.
 */
class TanSigmoid : public Kernel {
public:
    /*
     * Function that computes the tan-sigmoid function value at the given 'x'.
     *
     * @param x Input value.
     * @return The value of the tan-sigmoid function at the given x value.
     */
    double kernelFunc(const double &x);

    /*
     * Function that computes the tan-sigmoid derivative value at the given 'x'.
     *
     * @param x Input value.
     * @return The derivative of the tan-sigmoid function at the given x value.
     */
    double kernelDeriv(const double &x);
};

#endif //TAN_SIGMOID_H

