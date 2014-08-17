#ifndef KERNEL_H
#define KERNEL_H

#include <cstddef>

using namespace std;

/**
 * Interface for a kernel.
 */
class Kernel {
public:
    /*
     * Function that computes the kernel function value at the given 'x'.
     *
     * @param x Input value.
     * @return The value of the kernel function at the given x value.
     */
    static double kernelFunc(const double &x) = 0;

    /*
     * Function that computes the kernel derivative value at the given 'x'.
     *
     * @param x Input value.
     * @return The derivative of the kernel function at the given x value.
     */
    static double kernelDeriv(const double &x) = 0;
};

#endif //KERNEL_H
