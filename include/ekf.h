#ifndef EKF_H_
#define EKF_H_

#include "kalman_filter.h"
#include "scenarios.h"

class ExtendedKalmanFilter : public KalmanFilter {
public:
    ExtendedKalmanFilter(int stateSize, int measurementSize, int inputSize)
        : KalmanFilter(stateSize, measurementSize, inputSize) {
    }

    ExtendedKalmanFilter(int stateSize, int measurementSize, int inputSize,
                    const Eigen::VectorXd& initState,
                    const Eigen::MatrixXd& initEstimateCovariance)
        : KalmanFilter(stateSize, measurementSize, inputSize, initState, initEstimateCovariance) {
    }

    void Predict(const Eigen::VectorXd& input = Eigen::VectorXd::Zero(0)) override {
        auto state = GetState();
        auto estimateCovariance = GetEstimateCovariance();

        Eigen::MatrixXd Fx = scenario_->GetFx(state, stateSize_);
        Eigen::MatrixXd dFx = scenario_->GetdFx(state, stateSize_);

        state = Fx;
        SetState(state);

        estimateCovariance = dFx * estimateCovariance * dFx.transpose() + processNoiseCovariance_;
        SetEstimateCovariance(estimateCovariance);
    }

    void Update(const Eigen::VectorXd& measurement) override {
        auto state = GetState();
        auto estimateCovariance = GetEstimateCovariance();

        Eigen::MatrixXd Hx = scenario_->GetHx(state, stateSize_, measurementSize_);
        Eigen::MatrixXd dHx = scenario_->GetdHx(state, stateSize_, measurementSize_);

        auto PHt = estimateCovariance * dHx.transpose();
        auto K = PHt * (dHx * PHt + measurementCovariance_).inverse();

        // std::cout << K << "\n------------\n";

        auto innovation = measurement - Hx;
        state = state + K * innovation;
        SetState(state);

        auto I = Eigen::MatrixXd::Identity(stateSize_, stateSize_);
        auto I_KH = I - K * dHx;
        estimateCovariance = I_KH * estimateCovariance * I_KH.transpose() + K * measurementCovariance_ * K.transpose();
        SetEstimateCovariance(estimateCovariance);
    }
};

#endif // EKF_H_