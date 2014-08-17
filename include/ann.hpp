#ifndef ANN_H
#define ANN_H

#include <cstddef>

using namespace std;

/**
 * Class for training and using an Artifical Neural Network (ANN).
 */
class Ann {
public:
    static unsigned int* matrixToClass(double **matrix, const size_t &rows, const size_t &cols);
};

#endif //ANN_H

