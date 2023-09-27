#include <iostream>
#include <vector>

#include "lkf.h"
#include "helper_functions.h"

int main()
{
    // Set noisy measurements
    std::vector<double> alt {6.43, 1.3, 39.43, 45.89, 41.44, 48.7, 78.06, 80.08, 61.77, 75.15, 110.9, 127.83, 158.75, 156.55, 213.32, 229.82, 262.8,
                            297.57, 335.69, 367.92, 377.19, 411.18, 460.7, 468.39, 553.9, 583.97, 655.15, 723.09, 736.85, 787.22};
    std::vector<double> acc {39.81, 39.67, 39.81, 39.84, 40.05, 39.85, 39.78, 39.65, 39.67, 39.78, 39.59, 39.87, 39.85, 39.59, 39.84, 39.9, 39.63,
                            39.59, 39.76, 39.79, 39.73, 39.93, 39.83, 39.85, 39.94, 39.86, 39.76, 39.86, 39.74, 39.94};

    // Init Kalman Filter

    int stateSize = 2;
    int measurementSize = 1;
    int inputSize = 1;

    Eigen::VectorXd initState = Eigen::VectorXd::Zero(stateSize);
    
    Eigen::MatrixXd initEstimateCovariance = Eigen::MatrixXd::Zero(stateSize, stateSize);
    initEstimateCovariance(0, 0) = 500;
    initEstimateCovariance(1, 1) = 500;

    LinearKalmanFilter LKF(stateSize, measurementSize, inputSize, initState, initEstimateCovariance);

    Eigen::MatrixXd stateTransitionMatrix = Eigen::MatrixXd(stateSize, stateSize);
    stateTransitionMatrix << 1, 0.25, 0, 1;
    LKF.SetStateTransitionMatrix(stateTransitionMatrix);

    Eigen::MatrixXd observationMatrix = Eigen::MatrixXd(inputSize, stateSize);
    observationMatrix << 1, 0;
    LKF.SetObservationMatrix(observationMatrix);

    Eigen::MatrixXd controlMatrix = Eigen::MatrixXd(stateSize, inputSize);
    controlMatrix << 0.0313, 0.25;
    LKF.SetControlMatrix(controlMatrix);

    const double dt = 0.25;
    const double var_acc = 0.1*0.1;
    Eigen::MatrixXd processNoiseCovariance = Eigen::MatrixXd(stateSize, stateSize);
    processNoiseCovariance << pow(dt,4)*var_acc/4.0, pow(dt,3)*var_acc/2.0, pow(dt,3)*var_acc/2.0, pow(dt,2)*var_acc;
    LKF.SetProcessNoiseCovariance(processNoiseCovariance);

    Eigen::MatrixXd measurementCovariance = Eigen::MatrixXd(measurementSize, measurementSize);
    measurementCovariance << 400;
    LKF.SetMeasurementCovariance(measurementCovariance);

    Eigen::MatrixXd estimatedState = Eigen::MatrixXd(alt.size(), stateSize);

    Eigen::VectorXd measurement = Eigen::VectorXd(measurementSize);
    Eigen::VectorXd input = Eigen::VectorXd(inputSize);
    input << 0.0;
    for (int n = 0; n < alt.size(); n++) {
        double alt_n = alt[n];
        double acc_n = acc[n];

        measurement(0) = alt_n;

        LKF.Predict(input);
        LKF.Update(measurement);
        input(0) = acc_n;

        auto state = LKF.GetState();

        estimatedState(n, 0) = state(0);
        estimatedState(n, 1) = state(1);
    }

    // Log data
    FileHandler fileHandler("state", estimatedState, {"altitude", "velocity"});
    fileHandler.AppendData();

    // std::cout << estimatedState << std::endl;
}