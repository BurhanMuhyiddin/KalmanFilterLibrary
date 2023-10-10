#include <iostream>
#include <vector>
#include "scenarios.h"

#include "lkf.h"
#include "helper_functions.h"

namespace {
    constexpr int SCENARIO = 1;
}

int main()
{
    Scenario1& scenario1 = Scenario1::GetInstance();

    // Read data
    auto gt_data = scenario1.GetGtData();

    auto meas_data = scenario1.GetMeasData(gt_data);

    auto inp_data = scenario1.GetInpData(gt_data);

    // Create KF instance and initialize
    int stateSize = scenario1.kf_params_.state_dim;
    int measurementSize = scenario1.kf_params_.meas_dim;
    int inputSize = scenario1.kf_params_.inp_dim;

    Eigen::VectorXd initState = scenario1.GetInitState(stateSize);
    
    Eigen::MatrixXd initEstimateCovariance = scenario1.GetInitEstimateCovariance(stateSize);

    LinearKalmanFilter LKF(stateSize, measurementSize, inputSize, initState, initEstimateCovariance);

    LKF.SetScenario(&scenario1);

    Eigen::MatrixXd stateTransitionMatrix = scenario1.GetF(stateSize);
    LKF.SetStateTransitionMatrix(stateTransitionMatrix);

    Eigen::MatrixXd observationMatrix = scenario1.GetH(stateSize, inputSize);
    LKF.SetObservationMatrix(observationMatrix);

    Eigen::MatrixXd controlMatrix = scenario1.GetG(stateSize, inputSize);
    LKF.SetControlMatrix(controlMatrix);

    Eigen::MatrixXd processNoiseCovariance = scenario1.GetProcessNoiseCovariance(stateSize);
    LKF.SetProcessNoiseCovariance(processNoiseCovariance);

    Eigen::MatrixXd measurementCovariance = scenario1.GetMeasurementCovariance(measurementSize);
    LKF.SetMeasurementCovariance(measurementCovariance);

    // Buffers to save intermediate data
    int num_time_steps = meas_data.first.rows();
    Eigen::MatrixXd estimatedStateBuffer = Eigen::MatrixXd(num_time_steps, stateSize);
    Eigen::MatrixXd estimateCovarianceBuffer = Eigen::MatrixXd(num_time_steps*stateSize, stateSize);

    Eigen::VectorXd measurement = Eigen::VectorXd(measurementSize);
    Eigen::VectorXd input = Eigen::VectorXd(inputSize);
    input << 0.0;
    for (int n = 0; n < num_time_steps; n++) {
        double alt_n = meas_data.first(n);
        double acc_n = inp_data.first(n);

        LKF.Predict(input);
        input(0) = acc_n;

        measurement(0) = alt_n;
        LKF.Update(measurement);

        auto state = LKF.GetState();
        auto estimateCovariance = LKF.GetEstimateCovariance();

        estimatedStateBuffer(n, 0) = state(0);
        estimatedStateBuffer(n, 1) = state(1);

        estimateCovarianceBuffer.block(stateSize*n, 0, stateSize, stateSize) = estimateCovariance;
    }

    // Log data
    DataLogger stateDataLogger("../data/LKF","state", estimatedStateBuffer, {stateSize, 1}, {"altitude", "velocity"});
    stateDataLogger.AppendData();

    DataLogger estimateCovarianceDataLogger("../data/LKF", "estimateCovariance", estimateCovarianceBuffer, {stateSize, stateSize});
    estimateCovarianceDataLogger.AppendData();

    DataLogger gtAltitudeDataLogger("../data/LKF", "gtAltitude", gt_data.first.col(0), {num_time_steps, 1}, {"gtAltitude"});
    gtAltitudeDataLogger.AppendData();

    DataLogger gtVelocityDataLogger("../data/LKF", "gtVelocity", gt_data.first.col(1), {num_time_steps, 1}, {"gtVelocity"});
    gtVelocityDataLogger.AppendData();

    DataLogger measAltitudeDataLogger("../data/LKF", "measAltitude", meas_data.first, {num_time_steps, 1}, {"measAltitude"});
    measAltitudeDataLogger.AppendData();
}