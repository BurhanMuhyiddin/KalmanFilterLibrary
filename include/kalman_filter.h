#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include <iostream>

#include <Eigen/Dense>

/**
 * @brief Base class for Kalman filters.
 * 
 * This class provides a common structure and interface for various types of Kalman filters.
 */
class KalmanFilter {
public:
    /**
     * @brief Constructor to initialize Kalman Filter
     * 
     * @param stateSize             Size of the state vector
     * @param measurementSize       Size of the measurement vector
     * @param inputSize             Size of the input vector
     */
    KalmanFilter(int stateSize, int measurementSize, int inputSize)
        : stateSize_(stateSize), measurementSize_(measurementSize), inputSize_(inputSize) {
        state_.resize(stateSize);
        estimateCovariance_.resize(stateSize, stateSize);
        measurementCovariance_.resize(measurementSize, measurementSize);
        processNoiseCovariance_.resize(stateSize, stateSize);
    }

    KalmanFilter(int stateSize,
                int measurementSize,
                int inputSize,
                const Eigen::VectorXd& initState,
                const Eigen::MatrixXd& initEstimateCovariance)
        : stateSize_(stateSize), measurementSize_(measurementSize), inputSize_(inputSize),
          state_(initState), estimateCovariance_(initEstimateCovariance) {
        measurementCovariance_.resize(measurementSize, measurementSize);
        processNoiseCovariance_.resize(stateSize, stateSize);
    }

    /**
     * @brief Virtual destructor for the Kalman filter.
     */
    virtual ~KalmanFilter() = default;

    void InitFilter(const Eigen::VectorXd& initState,
                    const Eigen::MatrixXd& initEstimateCovariance) {
        state_ = initState;
        estimateCovariance_ = initEstimateCovariance;
    }

    /**
     * @brief Prediction step of the Kalman filter.
     * 
     * This method is responsible for predicting the next state and state covariance.
     * 
     * @param input     Input to the system. Default is zero.
     */
    virtual void Predict(const Eigen::VectorXd& input = Eigen::VectorXd::Zero(0)) = 0;

    /**
     * @brief Update step of the Kalman filter
     * 
     * @param measurement     The measurement vector used to update state and state covariance.
     */
    virtual void Update(const Eigen::VectorXd& measurement)  = 0;

    void SetState(const Eigen::VectorXd& state) {
        state_ = state;
    }

    void SetEstimateCovariance(const Eigen::MatrixXd& estimateCovariance) {
        estimateCovariance_ = estimateCovariance;
    }

    void SetMeasurementCovariance(const Eigen::MatrixXd& measurementCovariance) {
        measurementCovariance_ = measurementCovariance;
    }

    void SetProcessNoiseCovariance(const Eigen::MatrixXd& processNoiseCovariance) {
        processNoiseCovariance_ = processNoiseCovariance;
    }

    /**
     * @brief Get the current state estimate.
     * 
     * @return The current state estimate as a vector. 
     */
    Eigen::VectorXd GetState() const {
        return state_;
    }

    /**
     * @brief Get the current state covariance matrix.
     * 
     * @return The current state covariance matrix.
     */
    Eigen::MatrixXd GetEstimateCovariance() {
        return estimateCovariance_;
    }

protected:
    int stateSize_;                                 /**< Size of the state vector */
    int measurementSize_;                           /**< Size of the measurement vector */
    int inputSize_;                                 /**< Size of the input vector */

    Eigen::VectorXd state_;                         /**< State estimate [x] */
    Eigen::MatrixXd estimateCovariance_;            /**< Estimate covariance matrix [P] */
    Eigen::MatrixXd measurementCovariance_;         /**< Measurement covariance [R]*/
    Eigen::MatrixXd processNoiseCovariance_;        /**< Process noise covariance [Q] */
};

#endif // KALMAN_FILTER_H_