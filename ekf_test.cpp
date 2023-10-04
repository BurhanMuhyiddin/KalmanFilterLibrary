#include <iostream>
#include <vector>
#include "scenarios.h"

#include "ekf.h"
#include "helper_functions.h"

int main()
{
    Scenario2& scenario2 = Scenario2::GetInstance();

    // Read data
    auto gtData = scenario2.GetGtData();

    auto measData = scenario2.GetMeasData(gtData);

    // Create KF instance and initialize
    int stateSize = scenario2.kf_params_.state_dim;
    int measurementSize = scenario2.kf_params_.meas_dim;
    int inputSize = scenario2.kf_params_.inp_dim;

    Eigen::VectorXd initState = scenario2.GetInitState(stateSize);
    
    Eigen::MatrixXd initEstimateCovariance = scenario2.GetInitEstimateCovariance(stateSize);

    ExtendedKalmanFilter EKF(stateSize, measurementSize, inputSize, initState, initEstimateCovariance);

    auto stateTransitionMatrix = scenario2.GetFMatrix(stateSize);
    EKF.SetStateTransitionMatrix(stateTransitionMatrix);

    auto observationMatrix = scenario2.GetHMatrix(initState, stateSize, measurementSize);
    EKF.SetObservationMatrix(observationMatrix);

    // Eigen::MatrixXd controlMatrix = scenario1.GetControlMatrix(stateSize, inputSize);
    // LKF.SetControlMatrix(controlMatrix);

    Eigen::MatrixXd processNoiseCovariance = scenario2.GetProcessNoiseCovariance(stateSize);
    EKF.SetProcessNoiseCovariance(processNoiseCovariance);

    Eigen::MatrixXd measurementCovariance = scenario2.GetMeasurementCovariance(measurementSize);
    EKF.SetMeasurementCovariance(measurementCovariance);

    // Buffers to save intermediate data
    Eigen::MatrixXd estimatedStateBuffer = Eigen::MatrixXd(measData.R.size(), stateSize);
    Eigen::MatrixXd estimateCovarianceBuffer = Eigen::MatrixXd(measData.R.size()*stateSize, stateSize);

    Eigen::VectorXd measurement = Eigen::VectorXd(measurementSize);

    for (int n = 0; n < measData.R.size(); n++) {
        double R_n = measData.R(n);
        double phi_n = measData.phi(n);

        measurement(0) = R_n;
        measurement(1) = phi_n;

        EKF.Predict();
        // std::cout << measurement << "\n-------\n";
        EKF.Update(measurement);

        auto state = EKF.GetState();
        auto estimateCovariance = EKF.GetEstimateCovariance();
        // std::cout << "After update: " << state << "\n-------\n";

        for (int i = 0; i < stateSize; i++) {
            estimatedStateBuffer(n, i) = state(i);
        }

        estimateCovarianceBuffer.block(stateSize*n, 0, stateSize, stateSize) = estimateCovariance;
    }

    // Log data
    DataLogger stateDataLogger("EKF/state", estimatedStateBuffer, {stateSize, 1}, {"x", "vx", "ax", "y", "vy", "ay"});
    stateDataLogger.AppendData();

    DataLogger estimateCovarianceDataLogger("EKF/estimateCovariance", estimateCovarianceBuffer, {stateSize, stateSize});
    estimateCovarianceDataLogger.AppendData();

    DataLogger gtStateLogger("EKF/gtState", gtData.X, {stateSize, 1}, {"gtX", "gtVx", "gtAx", "gtY", "gtVy", "gtAy"});
    gtStateLogger.AppendData();

    return 0;
}