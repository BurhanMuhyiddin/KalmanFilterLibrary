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

    Eigen::MatrixXd stateTransitionMatrix = scenario1.GetStateTransitionMatrix(stateSize);
    LKF.SetStateTransitionMatrix(stateTransitionMatrix);

    Eigen::MatrixXd observationMatrix = scenario1.GetObservationMatrix(stateSize, inputSize);
    LKF.SetObservationMatrix(observationMatrix);

    Eigen::MatrixXd controlMatrix = scenario1.GetControlMatrix(stateSize, inputSize);
    LKF.SetControlMatrix(controlMatrix);

    Eigen::MatrixXd processNoiseCovariance = scenario1.GetProcessNoiseCovariance(stateSize);
    LKF.SetProcessNoiseCovariance(processNoiseCovariance);

    Eigen::MatrixXd measurementCovariance = scenario1.GetMeasurementCovariance(measurementSize);
    LKF.SetMeasurementCovariance(measurementCovariance);

    // Buffers to save intermediate data
    Eigen::MatrixXd estimatedStateBuffer = Eigen::MatrixXd(meas_data.alt.size(), stateSize);
    Eigen::MatrixXd estimateCovarianceBuffer = Eigen::MatrixXd(meas_data.alt.size()*stateSize, stateSize);

    Eigen::VectorXd measurement = Eigen::VectorXd(measurementSize);
    Eigen::VectorXd input = Eigen::VectorXd(inputSize);
    input << 0.0;
    for (int n = 0; n < meas_data.alt.size(); n++) {
        double alt_n = meas_data.alt(n);
        double acc_n = inp_data.acc(n);

        measurement(0) = alt_n;

        LKF.Predict(input);
        LKF.Update(measurement);
        input(0) = acc_n;

        auto state = LKF.GetState();
        auto estimateCovariance = LKF.GetEstimateCovariance();

        estimatedStateBuffer(n, 0) = state(0);
        estimatedStateBuffer(n, 1) = state(1);

        estimateCovarianceBuffer.block(stateSize*n, 0, stateSize, stateSize) = estimateCovariance;
    }

    // Log data
    DataLogger stateDataLogger("state", estimatedStateBuffer, {stateSize, 1}, {"altitude", "velocity"});
    stateDataLogger.AppendData();

    DataLogger estimateCovarianceDataLogger("estimateCovariance", estimateCovarianceBuffer, {stateSize, stateSize});
    estimateCovarianceDataLogger.AppendData();

    DataLogger gtAltitudeDataLogger("gtAltitude", gt_data.alt, {gt_data.alt.rows(), gt_data.alt.cols()}, {"gtAltitude"});
    gtAltitudeDataLogger.AppendData();

    DataLogger gtVelocityDataLogger("gtVelocity", gt_data.vel, {gt_data.vel.rows(), gt_data.vel.cols()}, {"gtVelocity"});
    gtVelocityDataLogger.AppendData();

    DataLogger measAltitudeDataLogger("measAltitude", meas_data.alt, {meas_data.alt.rows(), meas_data.alt.cols()}, {"measAltitude"});
    measAltitudeDataLogger.AppendData();
}