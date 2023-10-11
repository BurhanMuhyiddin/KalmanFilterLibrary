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

        sigma_points_ = Eigen::MatrixXd(stateSize_, num_sigma_points_);

        CalculateWeights();
    }

    UnscentedKalmanFilter(int stateSize, int measurementSize, int inputSize,
                    const Eigen::VectorXd& initState,
                    const Eigen::MatrixXd& initEstimateCovariance)
        : KalmanFilter(stateSize, measurementSize, inputSize, initState, initEstimateCovariance) {
        N_ = stateSize_;
        num_sigma_points_ = 2 * N_ + 1;
        kappa_ = 3 - N_;

        sigma_points_ = Eigen::MatrixXd(stateSize_, num_sigma_points_);

        CalculateWeights();
    }

    void Predict(const Eigen::VectorXd& input = Eigen::VectorXd::Zero(0)) override {
        // Calculate sigma points
        CalculateSigmaPoints();

        // Propogate sigma points
        propogated_sigma_points_ = scenario_->GetFx(sigma_points_, stateSize_);

        // Calculate sample mean and covariance
        auto sample_mean_and_covariance = CalculateSampleMeanAndCovariance(propogated_sigma_points_);

        // Set state and covariance matrix
        SetState(sample_mean_and_covariance.first);
        SetEstimateCovariance(sample_mean_and_covariance.second + processNoiseCovariance_);
    }

    void Update(const Eigen::VectorXd& measurement) override {
        auto state = GetState();
        auto estimate_covariance = GetEstimateCovariance();

        // Transfer from state space to measurement space
        Eigen::MatrixXd Hx = scenario_->GetHx(propogated_sigma_points_, stateSize_, measurementSize_);

        // Calculate sample mean and covariance
        auto sample_mean_and_covariance = CalculateSampleMeanAndCovariance(Hx);
        sample_mean_and_covariance.second += measurementCovariance_;

        // Calculate cross covariance
        auto cross_covariance = CalculateCrossCovariance(Hx, sample_mean_and_covariance.first, propogated_sigma_points_, state_);

        // Calculate Kalman Gain
        auto K = cross_covariance * sample_mean_and_covariance.second.inverse();

        // Update state and estimate covariance
        auto innovation_term = measurement - sample_mean_and_covariance.first;
        SetState(state + K * innovation_term);
        auto KPKt = K * sample_mean_and_covariance.second * K.transpose();
        SetEstimateCovariance(estimate_covariance - KPKt);
    }

public:
    void CalculateWeights() {
        double w0 = kappa_ / (N_ + kappa_);
        double w1 = 1.0 / (2.0 * (N_ + kappa_));

        Eigen::VectorXd weights(num_sigma_points_);
        weights(0) = w0;
        for (int i = 1; i < num_sigma_points_; i++) {
            weights(i) = w1;
        }

        weights_ = weights;
        weights_diag_ = weights.matrix().asDiagonal();
    }

    void CalculateSigmaPoints() {
        Eigen::VectorXd state = GetState();
        Eigen::MatrixXd state_covariance = GetEstimateCovariance();

        // Calculate square root of (N + k)*state_covariance
        Eigen::MatrixXd scaled_state_covariance = (N_ + kappa_) * state_covariance;
        // Eigen::MatrixXd sqrt_state_covariance = scaled_state_covariance.sqrt();
        Eigen::MatrixXd sqrt_state_covariance = scaled_state_covariance.llt().matrixL();

        sigma_points_.col(0) = state;
        for (int i = 0; i < state_covariance.cols(); i++) {
            // get column i
            Eigen::VectorXd column_i = sqrt_state_covariance.col(i);

            sigma_points_.col(i+1) = state + column_i;
            sigma_points_.col(stateSize_+i+1) = state - column_i;
        }
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> CalculateSampleMeanAndCovariance(const Eigen::MatrixXd& inp_matrix) const {
        Eigen::VectorXd sample_mean = inp_matrix * weights_;

        Eigen::MatrixXd deviation(inp_matrix);
        for (int i = 0; i < deviation.cols(); i++) {
            deviation.col(i) -= sample_mean;
        }
        Eigen::MatrixXd sample_covariance = deviation * weights_diag_ * deviation.transpose();
    
        return {sample_mean, sample_covariance};
    }

    Eigen::MatrixXd CalculateCrossCovariance(const Eigen::MatrixXd& Hx, const Eigen::VectorXd& Hx_mean,
                                            const Eigen::MatrixXd& Fx, const Eigen::VectorXd& Fx_mean) const {
        Eigen::MatrixXd deviation_Fx(Fx);
        for (int i = 0; i < deviation_Fx.cols(); i++) {
            deviation_Fx.col(i) -= Fx_mean;
        }

        Eigen::MatrixXd deviation_Hx(Hx);
        for (int i = 0; i < deviation_Hx.cols(); i++) {
            deviation_Hx.col(i) -= Hx_mean;
        }

        Eigen::MatrixXd cross_covariance = deviation_Fx * weights_diag_ * deviation_Hx.transpose();

        return cross_covariance;
    }

private:
    int num_sigma_points_;
    int N_;
    int kappa_;

    Eigen::VectorXd weights_;
    Eigen::MatrixXd weights_diag_;
    Eigen::MatrixXd sigma_points_;
    Eigen::MatrixXd propogated_sigma_points_;
};

#endif // UKF_H_