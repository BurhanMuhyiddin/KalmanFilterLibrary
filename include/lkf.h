#include "kalman_filter.h"

class LinearKalmanFilter : public KalmanFilter {
public:
    LinearKalmanFilter(int stateSize, int measurementSize, int inputSize)
        : KalmanFilter(stateSize, measurementSize, inputSize) {
    }

    LinearKalmanFilter(int stateSize, int measurementSize, int inputSize,
                    const Eigen::VectorXd& initState,
                    const Eigen::MatrixXd& initEstimateCovariance)
        : KalmanFilter(stateSize, measurementSize, inputSize, initState, initEstimateCovariance) {
    }

    void Predict(const Eigen::VectorXd& input = Eigen::VectorXd::Zero(0)) override {
        auto state = GetState();
        auto estimateCovariance = GetEstimateCovariance();

        state = stateTransitionMatrix_ * state + controlMatrix_ * input;
        SetState(state);

        estimateCovariance = stateTransitionMatrix_ * estimateCovariance * stateTransitionMatrix_.transpose() + processNoiseCovariance_;
        SetEstimateCovariance(estimateCovariance);
    }

    void Update(const Eigen::VectorXd& measurement) override {
        auto state = GetState();
        auto estimateCovariance = GetEstimateCovariance();

        // std::cout << "observationMatrix_: " << observationMatrix_.size() << "\n";
        // std::cout << "estimateCovariance: " << estimateCovariance.size() << "\n";

        auto HPHt = observationMatrix_ * estimateCovariance * observationMatrix_.transpose();
        auto K = estimateCovariance * observationMatrix_.transpose() *
                (HPHt + measurementCovariance_).inverse();

        auto innovation = measurement - observationMatrix_ * state;
        state = state + K * innovation;
        SetState(state);

        auto I = Eigen::MatrixXd::Identity(stateSize_, stateSize_);
        auto I_KH = I - K * observationMatrix_;
        estimateCovariance = I_KH * estimateCovariance * 
                            I_KH.transpose() + K * measurementCovariance_ * K.transpose();
        SetEstimateCovariance(estimateCovariance);
    }

    void SetStateTransitionMatrix(const Eigen::MatrixXd& stateTransitionMatrix) {
        stateTransitionMatrix_ = stateTransitionMatrix;
    }

    void SetControlMatrix(const Eigen::MatrixXd& controlMatrix) {
        controlMatrix_ = controlMatrix;
    }

    void SetObservationMatrix(const Eigen::MatrixXd& observationMatrix) {
        observationMatrix_ = observationMatrix;
    }

private:
    Eigen::MatrixXd stateTransitionMatrix_;     /**< State transition matrix [F] */
    Eigen::MatrixXd controlMatrix_;             /**< Control matrix [G] */
    Eigen::MatrixXd observationMatrix_;         /**< Observation matrix [H] */
};