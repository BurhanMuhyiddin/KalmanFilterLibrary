#ifndef EKF_H_
#define EKF_H_

#include "kalman_filter.h"

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

        Eigen::MatrixXd Fx = stateTransitionMatrix_.first;
        Eigen::MatrixXd dFx = stateTransitionMatrix_.second;

        state = Fx * state;
        SetState(state);

        // std::cout << Fx.rows() << "," << Fx.cols() << "\n";
        // std::cout << dFx.rows() << "," << dFx.cols() << "\n";
        // std::cout << estimateCovariance.rows() << "," << estimateCovariance.cols() << "\n";
        // std::cout << processNoiseCovariance_.rows() << "," << processNoiseCovariance_.cols() << "\n";

        estimateCovariance = dFx * estimateCovariance * dFx.transpose() + processNoiseCovariance_;
        SetEstimateCovariance(estimateCovariance);
    }

    void Update(const Eigen::VectorXd& measurement) override {
        auto state = GetState();
        auto estimateCovariance = GetEstimateCovariance();

        // std::cout << "observationMatrix_: " << observationMatrix_.size() << "\n";
        // std::cout << "estimateCovariance: " << estimateCovariance.size() << "\n";

        auto& scenario2 = Scenario2::GetInstance();
        auto observation_matrix = scenario2.GetHx(state, stateSize_, measurementSize_);

        Eigen::MatrixXd Hx = observation_matrix.first;
        Eigen::MatrixXd dHx = observation_matrix.second;

        // std::cout << "Hx: \n" << Hx << "\n";
        // std::cout << "dHx: \n" << dHx << "\n";

        auto PHt = estimateCovariance * dHx.transpose();
        auto K = PHt * (dHx * PHt + measurementCovariance_).inverse();

        // std::cout << K << "\n------------\n";

        auto innovation = measurement - Hx;
        // std::cout << "meas" <<  measurement << "\n------------\n";
        // std::cout << "Hx" <<  Hx << "\n------------\n";
        state = state + K * innovation;
        SetState(state);

        auto I = Eigen::MatrixXd::Identity(stateSize_, stateSize_);
        auto I_KH = I - K * dHx;
        estimateCovariance = I_KH * estimateCovariance * I_KH.transpose() + K * measurementCovariance_ * K.transpose();
        SetEstimateCovariance(estimateCovariance);
    }

    void SetStateTransitionMatrix(const std::pair<Eigen::MatrixXd, Eigen::MatrixXd>& stateTransitionMatrix) {
        stateTransitionMatrix_ = stateTransitionMatrix;
    }

    void SetControlMatrix(const Eigen::MatrixXd& controlMatrix) {
        controlMatrix_ = controlMatrix;
    }

    void SetObservationMatrix(const std::pair<Eigen::MatrixXd, Eigen::MatrixXd>& observationMatrix) {
        observationMatrix_ = observationMatrix;
    }

private:
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> stateTransitionMatrix_;     /**< State transition matrix [F] and its partial derivative [Fx] */
    Eigen::MatrixXd controlMatrix_;                                         /**< Control matrix [G] */
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> observationMatrix_;         /**< Observation matrix [H] and its derivative [Hx]*/
};

#endif // EKF_H_