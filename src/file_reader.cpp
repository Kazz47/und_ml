#include "file_reader.hpp"

#include <sstream>
#include <iterator>
#include <string>
#include <cassert>
#include <iostream>

#include "matrix_ops.hpp"

FileReader::FileReader(char *file_name) {
    try {
        infile.open(file_name);
    } catch (ifstream::failure e) {
        //TODO Log error!
        cout << "Exception opening/reading file." << endl;
    }

    if (!infile.is_open()) {
        //TODO Log file not found?
        cout << "The file '" << file_name << "' was not found." << endl;
    }
}

FileReader::~FileReader() {
    infile.close();
}

double** FileReader::getMatrix(size_t *rows, size_t *cols) {
    assert(infile.is_open());
    size_t num_rows = 0;
    size_t num_columns = 0;

    string line;
    while (getline(infile, line)) {
        num_rows++;

        istringstream iss(line);
        istream_iterator<double> eos;
        istream_iterator<double> it(iss);

        size_t col_count = 0;
        while (it != eos) {
            col_count++;
            it++;
        }

        if (num_columns == 0) {
            num_columns = col_count;
        } else if (col_count != num_columns) {
            //TODO Throw error and free matrix.
            return NULL;
        }
    }
    *rows = num_rows;
    *cols = num_columns;

    double **matrix = MatrixOps::newMatrix(*rows, *cols);

    infile.seekg(0);
    size_t r = 0;
    while (getline(infile, line)) {
        istringstream iss(line);
        istream_iterator<double> eos;
        istream_iterator<double> it(iss);

        size_t c = 0;
        while (it != eos) {
            matrix[r][c] = *it;
            c++;
            it++;
        }
        r++;
    }

    return matrix;
}
