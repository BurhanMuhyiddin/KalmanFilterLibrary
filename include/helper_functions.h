#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <Eigen/Dense>

class FileHandler {
public:
    FileHandler(const std::string& matrixName, const Eigen::MatrixXd& matrix, const std::vector<std::string>& columnNames = {})
        : matrixName_(matrixName), matrix_(matrix), columnNames_(columnNames) {
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

    ~FileHandler() {
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
        file_ << matrix_.rows() << "," << matrix_.cols() << "\n";
    }

    std::string matrixName_;
    std::string fileName_;
    Eigen::MatrixXd matrix_;
    std::vector<std::string> columnNames_;
    std::ofstream file_;
};