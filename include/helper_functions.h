#ifndef HELPER_FUNCTIONS_H_
#define HELPER_FUNCTIONS_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>

#include <Eigen/Dense>

class DataLogger {
public:
    DataLogger(const std::string& matrixName, const Eigen::MatrixXd& matrix, const std::pair<int, int>& dimension, const std::vector<std::string>& columnNames = {})
        : matrixName_(matrixName), matrix_(matrix), dimension_(dimension), columnNames_(columnNames) {
        fileName_ = "../data/" + matrixName_ + ".txt";
        file_.open(fileName_);

        if (!file_.is_open()) {
            std::cout << "Error: Unable to open file " << fileName_ << std::endl;
            return;
        }

        if (!columnNames_.empty()) {
            WriteColumnNames();
        }

        WriteMatrixDimensions();
    }

    ~DataLogger() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    void AppendData() {
        if (!file_.is_open()) {
            std::cerr << "Error: File " << fileName_ << " is not open." << std::endl;
            return;
        }

        file_ << "data\n";

        for (int i = 0; i < matrix_.rows(); i++) {
            for (int j = 0; j < matrix_.cols(); j++) {
                file_ << matrix_(i, j);
                if (j < matrix_.cols() - 1) {
                    file_ << ",";
                }
            }
            file_ << "\n";
        }

        std::cout << "Data appended to file " << fileName_ << std::endl;
    }

private:

    void WriteColumnNames() {
        file_ << "columns\n";

        for (const std::string& columnName : columnNames_) {
            file_ << columnName << ",";
        }
        file_ << "\n";
    }

    void WriteMatrixDimensions() {
        file_ << "dimension\n";
        file_ << dimension_.first << "," << dimension_.second << "\n";
    }

    std::string matrixName_;
    std::string fileName_;
    Eigen::MatrixXd matrix_;
    std::pair<int, int> dimension_;
    std::vector<std::string> columnNames_;
    std::ofstream file_;
};

// helper functions

Eigen::MatrixXd generate_noise(int nrows, int ncols, double mean=0.0, double sigma=1.0, int seed=-1) {
    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::normal_distribution<double> dis{0.0, 1.0};

    if (seed > 0)
        gen.seed(seed);

    Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(nrows, ncols).unaryExpr([&](float dummy){return dis(gen);});

    return noise;
}

void add_noise(Eigen::MatrixXd& x, const Eigen::VectorXd& sigma, int seed=-1) {

    Eigen::MatrixXd noise = generate_noise(x.rows(), x.cols(), 0.0, 1.0, seed);

    if (noise.cols() > noise.rows()) {
        noise = noise.transpose();
    }

    for (int i = 0; i < noise.cols(); i++) {
        noise.col(i) *= sigma(i);
    }

    if (noise.cols() > noise.rows()) {
        noise = noise.transpose();
    }

    x += noise;
}

#endif // HELPER_FUNCTIONS_H_