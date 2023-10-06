#ifndef SCENARIOS_H_
#define SCENARIOS_H_

#include <iostream>
#include <fstream>

#include <json.hpp>

#include <Eigen/Dense>

#include "helper_functions.h"

class DynamicSystem {
public:
    DynamicSystem() = default;
        
protected:
    virtual Eigen::MatrixXd GetProcessNoiseCovariance(int state_size) const = 0;
    virtual Eigen::MatrixXd GetMeasurementCovariance(int measurement_size) const = 0;
    virtual Eigen::VectorXd GetInitState(int state_size) const = 0;
    virtual Eigen::MatrixXd GetInitEstimateCovariance(int state_size) const = 0;

    void SetTimeStep(double dt) {
        dt_ = dt;
    }

protected:
    double dt_;
};

class Linear {
public:
    Linear() = default;
protected:
    virtual Eigen::MatrixXd GetF(int state_size) const = 0;
    virtual Eigen::MatrixXd GetH(int state_size, int input_size) const = 0;
    virtual Eigen::MatrixXd GetG(int state_size, int input_size) const = 0;
};

class NonLinear {
public:
    NonLinear() = default;
protected:
    virtual Eigen::MatrixXd GetFx() const = 0;
    virtual Eigen::MatrixXd GetHx() const = 0;
};

class BaseScenario {
protected:
    using DataType = std::pair<Eigen::MatrixXd, std::vector<std::string>>;
public:
    BaseScenario() = default;
protected:
    virtual DataType GetGtData() const = 0;
    virtual DataType GetMeasData(const DataType& gt_data) const = 0;
    virtual DataType GetInpData(const DataType& gt_data) const = 0;
};

class Scenario1 : public BaseScenario, public DynamicSystem, public Linear {
private:
    struct KF {
        int state_dim;
        int inp_dim;
        int meas_dim;
        double dt;
        std::vector<double> r;
        std::vector<double> sig_a;
        double g;
        std::string strModel;
        std::string strNoiseModel;
        std::vector<double> x0;
        std::vector<double> u0;
        std::vector<double> P0;
    };

    struct NoiseGenAlt {
        int seed;
    };

    struct NoiseGenAcc {
        int seed;
    };

    struct Scenario {
        bool isPlotScenario;
        double g;
        double x0;
        double v0;
        double a;
        int N;
        std::vector<double> sig_a;
        int seed;
    };

public:
    Scenario1() {
        std::ifstream f("../scenarios/scenario1.json");
        auto parameters = nlohmann::json::parse(f);

        kf_params_.state_dim = parameters["KF"]["state_dim"].get<int>();
        kf_params_.inp_dim = parameters["KF"]["inp_dim"].get<int>();
        kf_params_.meas_dim = parameters["KF"]["meas_dim"].get<int>();
        kf_params_.dt = parameters["KF"]["dt"].get<double>();
        kf_params_.g = parameters["KF"]["g"].get<double>();
        kf_params_.P0 = parameters["KF"]["P0"].get<std::vector<double>>();
        kf_params_.r = parameters["KF"]["r"].get<std::vector<double>>();
        kf_params_.sig_a = parameters["KF"]["sig_a"].get<std::vector<double>>();
        kf_params_.strModel = parameters["KF"]["strModel"].get<std::string>();
        kf_params_.strNoiseModel = parameters["KF"]["strNoiseModel"].get<std::string>();
        kf_params_.u0 = parameters["KF"]["u0"].get<std::vector<double>>();
        kf_params_.x0 = parameters["KF"]["x0"].get<std::vector<double>>();

        alt_noise_gen_params_.seed = parameters["noiseGenAlt"]["seed"].get<int>();
        
        acc_noise_gen_params_.seed = parameters["noiseGenAcc"]["seed"].get<int>();
        
        scenario_params_.a = parameters["scenario"]["a"].get<double>();
        scenario_params_.g = parameters["scenario"]["g"].get<double>();
        scenario_params_.isPlotScenario = parameters["scenario"]["isPlotScenario"].get<bool>();
        scenario_params_.N = parameters["scenario"]["N"].get<int>();
        scenario_params_.seed = parameters["scenario"]["seed"].get<int>();
        scenario_params_.sig_a = parameters["scenario"]["sig_a"].get<std::vector<double>>();
        scenario_params_.v0 = parameters["scenario"]["v0"].get<double>();
        scenario_params_.x0 = parameters["scenario"]["x0"].get<double>();

        SetTimeStep(kf_params_.dt);
    }

    Scenario1(const Scenario1&) = delete;
    Scenario1(Scenario1&&) = delete;
    Scenario1& operator=(const Scenario1&) = delete;
    Scenario1& operator=(Scenario1&&) = delete;

public:
    Eigen::MatrixXd GetProcessNoiseCovariance(int state_size) const override {
        const double dt = dt_;
        const double sig_a = scenario_params_.sig_a[0];
        const double var_acc = sig_a * sig_a;
        Eigen::MatrixXd process_noise_covariance = Eigen::MatrixXd(state_size, state_size);
        process_noise_covariance << pow(dt,4)*var_acc/4.0, pow(dt,3)*var_acc/2.0, pow(dt,3)*var_acc/2.0, pow(dt,2)*var_acc;

        return process_noise_covariance;
    }

    Eigen::MatrixXd GetMeasurementCovariance(int measurement_size) const override {
        const std::vector<double> sigma_meas = kf_params_.r;

        Eigen::MatrixXd measurement_covariance = Eigen::MatrixXd::Identity(measurement_size, measurement_size);
        for (int i = 0; i < sigma_meas.size(); i++) {
            measurement_covariance(i, i) *= pow(sigma_meas[i], 2);
        }

        return measurement_covariance;
    }

    Eigen::VectorXd GetInitState(int state_size) const override {
        Eigen::VectorXd init_state = Eigen::VectorXd::Zero(state_size);
        for (int i = 0; i < state_size; i++) {
            init_state(i) = kf_params_.x0[i];
        }

        return init_state;
    }

    Eigen::MatrixXd GetInitEstimateCovariance(int state_size) const override {
        Eigen::MatrixXd init_estimate_covariance = Eigen::MatrixXd::Zero(state_size, state_size);
        for (int i = 0; i < state_size; i++) {
            init_estimate_covariance(i, i) = kf_params_.P0[i];
        }

        return init_estimate_covariance;
    }

    DataType GetGtData() const override {
        double x0 = scenario_params_.x0;
        double v0 = scenario_params_.v0;
        double a = scenario_params_.a;
        int N = scenario_params_.N;
        double g = scenario_params_.g;
        std::vector<double> sig_a = scenario_params_.sig_a;

        double start = 0;
        double end = (double)(dt_*N);
        double step_size = dt_;

        // int num_points = static_cast<int>((end - start) / step_size) + 1;
        Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(N, start, end); // column vector (N,1)

        Eigen::MatrixXd a_vec(t.rows(), 1);
        a_vec.setConstant(a);

        add_noise(a_vec, sig_a);

        Eigen::VectorXd x = Eigen::VectorXd(t.rows()).setConstant(x0).array() + t.array() * v0 + 0.5 * a_vec.array() * t.array() * t.array();
        Eigen::VectorXd v = Eigen::VectorXd(t.rows()).setConstant(v0).array() + t.array() * a_vec.array();
        a_vec = a_vec.array() - g;

        Eigen::MatrixXd gt_data(t.rows(), 3);
        gt_data.col(0) = x;
        gt_data.col(1) = v;
        gt_data.col(2) = a_vec;

        return {gt_data, {"altitude", "velocity", "acc"}};
    }

    DataType GetMeasData(const DataType& gt_data) const override {
        auto r = kf_params_.r;

        std::vector<double> sig_a = scenario_params_.sig_a;
        Eigen::VectorXd sig_a_vec(sig_a.size());
        for (int i = 0; i < sig_a.size(); i++) {
            sig_a_vec[i] = sig_a[i];
        }

        int alt_noise_params = alt_noise_gen_params_.seed;
        Eigen::MatrixXd true_alt = gt_data.first.col(0);
        Eigen::MatrixXd Z(true_alt);
        add_noise(Z, r, alt_noise_params);

        return {Z, {"altitude"}};
    }

    DataType GetInpData(const DataType& gt_data) const override {
        std::vector<double> sig_a = scenario_params_.sig_a;
        
        int acc_noise_params = acc_noise_gen_params_.seed;
        Eigen::MatrixXd true_acc = gt_data.first.col(2);
        Eigen::MatrixXd A = true_acc;
        add_noise(A, sig_a, acc_noise_params);

        double g = kf_params_.g;
        Eigen::MatrixXd G(A);
        G.setConstant(g);
        Eigen::MatrixXd U = A + G;

        return {U, {"acc"}};
    }

    Eigen::MatrixXd GetF(int state_size) const override {
        Eigen::MatrixXd state_transition_matrix = Eigen::MatrixXd(state_size, state_size);
        state_transition_matrix << 1, 0.25, 0, 1;

        return state_transition_matrix;
    }

    Eigen::MatrixXd GetH(int state_size, int input_size) const override {
        Eigen::MatrixXd observation_matrix = Eigen::MatrixXd(input_size, state_size);
        observation_matrix << 1, 0;
        
        return observation_matrix;
    }

    Eigen::MatrixXd GetG(int state_size, int input_size) const override {
        Eigen::MatrixXd control_matrix = Eigen::MatrixXd(state_size, input_size);
        control_matrix << 0.0313, 0.25;
        
        return control_matrix;
    }

    static Scenario1& GetInstance() {
        static Scenario1 scenario1;
        return scenario1;
    }

public:
    KF kf_params_;
    NoiseGenAlt alt_noise_gen_params_;
    NoiseGenAcc acc_noise_gen_params_;
    Scenario scenario_params_;
};

// class Scenario2 : public Linear {
// public:
//     Scenario2() {}
// };

#endif // SCENARIOS_H_