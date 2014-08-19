#ifndef ANN_H
#define ANN_H

#include <cstddef>

using namespace std;

/**
 * Class for training and using an Artifical Neural Network (ANN).
 */
template <typename K>
class Ann {
public:

    Ann(const K kernel);
    //~Ann();

    //TODO Update this comment
    /**
     * Method to pass inputs through a ANN and get classification results.
     *
     * @param input Matrix of input values for the ANN.
     * @param weights Matrix of weights for the ANN nodes.
     * @param bias Matrix of bias for the ANN nodes.
     * @return The resulting classification matrix.
     */
    double** feedForward(
            double **input, const size_t &input_rows, const size_t &input_cols,
            double **weights, const size_t &weights_rows, const size_t &weights_cols,
            double **bias, const size_t &bias_rows, const size_t &bias_cols);

private:
    K kernel;
};

#endif //ANN_H

