#include <iostream>
#include <vector>
#include "scenarios.h"

#include "ekf.h"
#include "helper_functions.h"

int main()
{
    Scenario2& scenario2 = Scenario2::GetInstance();

    // Read data
    auto gt_data = scenario2.GetGtData();

    auto meas_data = scenario2.GetMeasData(gt_data);

    // Create KF instance and initialize
    int stateSize = scenario2.kf_params_.state_dim;
    int measurementSize = scenario2.kf_params_.meas_dim;
    int inputSize = scenario2.kf_params_.inp_dim;

    Eigen::VectorXd initState = scenario2.GetInitState(stateSize);
    
    Eigen::MatrixXd initEstimateCovariance = scenario2.GetInitEstimateCovariance(stateSize);

    ExtendedKalmanFilter EKF(stateSize, measurementSize, inputSize, initState, initEstimateCovariance);

    EKF.SetScenario(&scenario2);

    Eigen::MatrixXd processNoiseCovariance = scenario2.GetProcessNoiseCovariance(stateSize);
    EKF.SetProcessNoiseCovariance(processNoiseCovariance);

    Eigen::MatrixXd measurementCovariance = scenario2.GetMeasurementCovariance(measurementSize);
    EKF.SetMeasurementCovariance(measurementCovariance);

    // Buffers to save intermediate data
    int num_time_steps = meas_data.first.rows();
    Eigen::MatrixXd estimatedStateBuffer = Eigen::MatrixXd(num_time_steps, stateSize);
    Eigen::MatrixXd estimateCovarianceBuffer = Eigen::MatrixXd(num_time_steps*stateSize, stateSize);

    Eigen::VectorXd measurement = Eigen::VectorXd(measurementSize);

    for (int n = 0; n < num_time_steps; n++) {
        double R_n = meas_data.first(n, 0);
        double phi_n = meas_data.first(n, 1);

        measurement(0) = R_n;
        measurement(1) = phi_n;

        EKF.Predict();
        EKF.Update(measurement);

        auto state = EKF.GetState();
        auto estimateCovariance = EKF.GetEstimateCovariance();

        for (int i = 0; i < stateSize; i++) {
            estimatedStateBuffer(n, i) = state(i);
        }

        estimateCovarianceBuffer.block(stateSize*n, 0, stateSize, stateSize) = estimateCovariance;
    }

    // Log data
    DataLogger stateDataLogger("../data/EKF", "state", estimatedStateBuffer, {stateSize, 1}, {"x", "vx", "ax", "y", "vy", "ay"});
    stateDataLogger.AppendData();

    DataLogger estimateCovarianceDataLogger("../data/EKF", "estimateCovariance", estimateCovarianceBuffer, {stateSize, stateSize});
    estimateCovarianceDataLogger.AppendData();

    DataLogger gtStateLogger("../data/EKF", "gtState", gt_data.first, {gt_data.first.cols(), 1}, gt_data.second);
    gtStateLogger.AppendData();

    return 0;
}