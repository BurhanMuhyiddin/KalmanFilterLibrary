#include <iostream>
#include <vector>
#include "scenarios.h"

#include "ukf.h"
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

    UnscentedKalmanFilter UKF(stateSize, measurementSize, inputSize, initState, initEstimateCovariance);

    UKF.SetScenario(&scenario2);

    Eigen::MatrixXd processNoiseCovariance = scenario2.GetProcessNoiseCovariance(stateSize);
    UKF.SetProcessNoiseCovariance(processNoiseCovariance);

    Eigen::MatrixXd measurementCovariance = scenario2.GetMeasurementCovariance(measurementSize);
    UKF.SetMeasurementCovariance(measurementCovariance);

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

        UKF.Predict();
        UKF.Update(measurement);

        auto state = UKF.GetState();
        auto estimateCovariance = UKF.GetEstimateCovariance();

        for (int i = 0; i < stateSize; i++) {
            estimatedStateBuffer(n, i) = state(i);
        }

        estimateCovarianceBuffer.block(stateSize*n, 0, stateSize, stateSize) = estimateCovariance;
    }

    // Log data
    DataLogger stateDataLogger("../data/UKF", "state", estimatedStateBuffer, {stateSize, 1}, {"x", "vx", "ax", "y", "vy", "ay"});
    stateDataLogger.AppendData();

    DataLogger estimateCovarianceDataLogger("../data/UKF", "estimateCovariance", estimateCovarianceBuffer, {stateSize, stateSize});
    estimateCovarianceDataLogger.AppendData();

    DataLogger gtStateLogger("../data/UKF", "gtState", gt_data.first, {gt_data.first.cols(), 1}, gt_data.second);
    gtStateLogger.AppendData();

    return 0;
}