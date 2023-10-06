#ifndef UKF_H_
#define UKF_H_

#include "kalman_filter.h"

class UnscentedKalmanFilter : public KalmanFilter {
public:
    UnscentedKalmanFilter(int stateSize, int measurementSize, int inputSize)
        : KalmanFilter(stateSize, measurementSize, inputSize) {
            N_ = stateSize_;
            num_sigma_points_ = 2 * N_ + 1;
            kappa_ = 3 - N_;

            weights_ = Eigen::MatrixXd(num_sigma_points_, num_sigma_points_);
            sigma_points_ = Eigen::MatrixXd(stateSize_, num_sigma_points_);

    }

    UnscentedKalmanFilter(int stateSize, int measurementSize, int inputSize,
                    const Eigen::VectorXd& initState,
                    const Eigen::MatrixXd& initEstimateCovariance)
        : KalmanFilter(stateSize, measurementSize, inputSize, initState, initEstimateCovariance) {
    }

    void Predict(const Eigen::VectorXd& input = Eigen::VectorXd::Zero(0)) override {
        // Calculate sigma points
        CalculateSigmaPoints();

        // Propogate sigma points

    }

    void Update(const Eigen::VectorXd& measurement) override {
    }

public:
    void CalculateWeights() {
        double w0 = kappa_ / (N_ + kappa_);
        double w1 = 1.0 / (2.0 * (N_ + kappa_));

        Eigen::VectorXd weigths(num_sigma_points_);
        weigths(0) = w0;
        for (int i = 1; i < num_sigma_points_; i++) {
            weigths(i) = w1;
        }

        weights_ = weigths.matrix().asDiagonal();
    }

    void CalculateSigmaPoints() {
        Eigen::VectorXd state = GetState();
        Eigen::MatrixXd state_covariance = GetEstimateCovariance();

        // Calculate square root of (N + k)*state_covariance
        Eigen::MatrixXd scaled_state_covariance = (N_ + kappa_) * state_covariance;
        Eigen::MatrixXd sqrt_state_covariance = scaled_state_covariance.sqrt();

        sigma_points_.col(0) = state;
        for (int i = 0; i < state_covariance.cols(); i++) {
            // get column i
            Eigen::VectorXd column_i = sqrt_state_covariance.col(i);

            sigma_points_.col(i+1) = state + column_i;
            sigma_points_.col(stateSize_+i+1) = state - column_i;
        }
    }

    void CalculateSampleMeanAndCovariance() {

    }

private:
    int num_sigma_points_;
    int N_;
    int kappa_;

    Eigen::MatrixXd weights_;
    Eigen::MatrixXd sigma_points_;
};

#endif // UKF_H_