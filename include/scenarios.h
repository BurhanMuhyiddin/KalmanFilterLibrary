#ifndef SCENARIOS_H_
#define SCENARIOS_H_

#include <iostream>
#include <fstream>

#include <json.hpp>

#include <Eigen/Dense>

#include "helper_functions.h"

class Scenario1 {
private:
    struct GtData {
        Eigen::VectorXd alt;
        Eigen::VectorXd vel;
        Eigen::VectorXd acc;
    };

    struct MeasData {
        Eigen::VectorXd alt;
    };

    struct InpData {
        Eigen::VectorXd acc;
    };

    struct KF {
        int dim;
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

protected:
    Scenario1() {
        std::ifstream f("../scenarios/scenario1.json");
        auto parameters = nlohmann::json::parse(f);

        kf_params_.dim = parameters["KF"]["dim"].get<int>();
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

        dt_ = kf_params_.dt;
    }

    Scenario1(const Scenario1&) = delete;
    Scenario1(Scenario1&&) = delete;
    Scenario1& operator=(const Scenario1&) = delete;
    Scenario1& operator=(Scenario1&&) = delete;

public:
    GtData GetGtDataTemplate() const {
        GtData gtData;
        return gtData;
    }

    MeasData GetMeasDataTemplate() const {
        MeasData measData;
        return measData;
    }

    InpData GetInpDataTemplate() const {
        InpData inpData;
        return inpData;
    }

    void GetGtData(GtData& gt_data) const {
        double x0 = scenario_params_.x0;
        double v0 = scenario_params_.v0;
        double a = scenario_params_.a;
        int N = scenario_params_.N;
        double g = scenario_params_.g;
        std::vector<double> sig_a = scenario_params_.sig_a;
        
        // Eigen::VectorXd sig_a_vec(sig_a.size());
        // for (int i = 0; i < sig_a.size(); i++) {
        //     sig_a_vec[i] = sig_a[i];
        // }

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

        gt_data.alt = x;
        gt_data.vel = v;
        gt_data.acc = a_vec;
    }

    void GetMeasData(const GtData& gt_data, MeasData& meas_data) const {
        auto r = kf_params_.r;
        // Eigen::VectorXd r_vec(r.size());
        // for (int i = 0; i < r.size(); i++) {
        //     r_vec[i] = r[i];
        // }

        std::vector<double> sig_a = scenario_params_.sig_a;
        Eigen::VectorXd sig_a_vec(sig_a.size());
        for (int i = 0; i < sig_a.size(); i++) {
            sig_a_vec[i] = sig_a[i];
        }


        int alt_noise_params = alt_noise_gen_params_.seed;
        auto true_alt = gt_data.alt;
        auto Z = Eigen::MatrixXd(true_alt);
        add_noise(Z, r, alt_noise_params);

        meas_data.alt = Eigen::VectorXd(Z.col(0));
    }

    void GetInpData(const GtData& gt_data, InpData& inp_data) const {
        std::vector<double> sig_a = scenario_params_.sig_a;
        // Eigen::VectorXd sig_a_vec(sig_a.size());
        // for (int i = 0; i < sig_a.size(); i++) {
        //     sig_a_vec[i] = sig_a[i];
        // }
        
        int acc_noise_params = acc_noise_gen_params_.seed;
        auto true_acc = gt_data.acc;
        auto A = Eigen::MatrixXd(true_acc);
        add_noise(A, sig_a, acc_noise_params);

        double g = kf_params_.g;
        Eigen::MatrixXd G(A);
        G.setConstant(g);
        auto U = A + G;

        inp_data.acc = Eigen::VectorXd(U.col(0));
    }

    Eigen::MatrixXd GetStateTransitionMatrix(int state_size) const {
        Eigen::MatrixXd state_transition_matrix = Eigen::MatrixXd(state_size, state_size);
        state_transition_matrix << 1, 0.25, 0, 1;

        return state_transition_matrix;
    }

    Eigen::MatrixXd GetObservationMatrix(int state_size, int input_size) const {
        Eigen::MatrixXd observation_matrix = Eigen::MatrixXd(input_size, state_size);
        observation_matrix << 1, 0;
        
        return observation_matrix;
    }

    Eigen::MatrixXd GetControlMatrix(int state_size, int input_size) const {
        Eigen::MatrixXd control_matrix = Eigen::MatrixXd(state_size, input_size);
        control_matrix << 0.0313, 0.25;
        
        return control_matrix;
    }

    Eigen::MatrixXd GetProcessNoiseCovariance(int state_size) const {
        const double dt = dt_;
        const double sig_a = scenario_params_.sig_a[0];
        const double var_acc = sig_a * sig_a;
        Eigen::MatrixXd process_noise_covariance = Eigen::MatrixXd(state_size, state_size);
        process_noise_covariance << pow(dt,4)*var_acc/4.0, pow(dt,3)*var_acc/2.0, pow(dt,3)*var_acc/2.0, pow(dt,2)*var_acc;

        return process_noise_covariance;
    }

    Eigen::MatrixXd GetMeasurementCovariance(int measurement_size) const {
        const double sigma_meas = kf_params_.r[0];
        Eigen::MatrixXd measurement_covariance = Eigen::MatrixXd(measurement_size, measurement_size);
        measurement_covariance << sigma_meas * sigma_meas;

        return measurement_covariance;
    }

    Eigen::VectorXd GetInitState(int state_size) const {
        Eigen::VectorXd init_state = Eigen::VectorXd::Zero(state_size);
        for (int i = 0; i < state_size; i++) {
            init_state(i) = kf_params_.x0[i];
        }

        return init_state;
    }

    Eigen::MatrixXd GetInitEstimateCovariance(int state_size) const {
        Eigen::MatrixXd init_estimate_covariance = Eigen::MatrixXd::Zero(state_size, state_size);
        for (int i = 0; i < state_size; i++) {
            init_estimate_covariance(i, i) = kf_params_.P0[i];
        }

        return init_estimate_covariance;
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

private:
    double dt_;
};


class Scenario2 {
private:
    struct KF {
        int dim;
        double dt;
        std::vector<double> r_s;
        std::vector<double> r_m;
        std::vector<double> sig_a;

        std::string strModel;
        std::string strNoiseModel;

        std::vector<double> x0;
        std::vector<double> P0;
    };

    struct NoiseGen {
        int seed;
    };

    struct Scenario {
        bool isPlotScenario;
        double v;
        double L;
        double R;
        std::vector<double> sig_a;
        int seed;
    };

    struct GtData {
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        Eigen::VectorXd vx;
        Eigen::VectorXd vy;
        Eigen::VectorXd ax;
        Eigen::VectorXd ay;
        Eigen::VectorXd R;
        Eigen::VectorXd phi;

        Eigen::MatrixXd X;
    };

    struct MeasData {
        Eigen::VectorXd R;
        Eigen::VectorXd phi;
    };

    struct InpData {

    };

protected:
    Scenario2() {
        std::ifstream f("../scenarios/scenario2.json");
        auto parameters = nlohmann::json::parse(f);

        kf_params_.dim = parameters["KF"]["dim"].get<int>();
        kf_params_.dt = parameters["KF"]["dt"].get<double>();
        kf_params_.P0 = parameters["KF"]["P0"].get<std::vector<double>>();
        kf_params_.r_s = parameters["KF"]["r_s"].get<std::vector<double>>();
        kf_params_.r_m = parameters["KF"]["r_m"].get<std::vector<double>>();
        kf_params_.sig_a = parameters["KF"]["sig_a"].get<std::vector<double>>();
        kf_params_.strModel = parameters["KF"]["strModel"].get<std::string>();
        kf_params_.strNoiseModel = parameters["KF"]["strNoiseModel"].get<std::string>();
        kf_params_.x0 = parameters["KF"]["x0"].get<std::vector<double>>();

        noise_gen_params_.seed = parameters["noiseGen"]["seed"].get<int>();

        scenario_params_.isPlotScenario = parameters["scenario"]["isPlotScenario"].get<bool>();
        scenario_params_.L = parameters["scenario"]["L"].get<double>();
        scenario_params_.R = parameters["scenario"]["R"].get<double>();
        scenario_params_.seed = parameters["scenario"]["seed"].get<int>();
        scenario_params_.sig_a = parameters["scenario"]["sig_a"].get<std::vector<double>>();
        scenario_params_.v = parameters["scenario"]["v"].get<double>();
    }

    Scenario2(const Scenario2&) = delete;
    Scenario2(Scenario2&&) = delete;
    Scenario2& operator=(const Scenario2&) = delete;
    Scenario2& operator=(Scenario2&&) = delete;

public:
    void GetGtData(GtData& gtData) const {
        double v = scenario_params_.v;
        double L = scenario_params_.L;
        double R = scenario_params_.R;

        // First part of the trajectory: straight path
        double mX0 = R; // x start position
        double mY0 = -L; // y start position

        double T = L / v; // duration of path

        double start = 0;
        double end = T;
        double step_size = kf_params_.dt;
        int num_points = static_cast<int>((end - start) / step_size) + 1;
        Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(num_points, start, end); // column vector (num_points,1)

        Eigen::MatrixXd x1 = Eigen::MatrixXd::Zero(t.rows(), t.cols());
        x1 = x1.array() + mX0;

        Eigen::MatrixXd y1 = v * t.array();
        y1 = y1.array() + mY0;

        Eigen::MatrixXd Vx1 = Eigen::MatrixXd::Zero(x1.rows(), x1.cols());
        Eigen::MatrixXd Vy1 = v * Eigen::MatrixXd::Ones(y1.rows(), y1.cols());

        Eigen::MatrixXd Ax1 = Eigen::MatrixXd::Zero(x1.rows(), x1.cols());
        Eigen::MatrixXd Ay1 = Eigen::MatrixXd::Zero(y1.rows(), y1.cols());

        // Second part of the trajectory: turning manevour
        double omega = v / R;

        T = 0.5 * M_PI * R / v;

        start = 0;
        end = T;
        step_size = kf_params_.dt;
        num_points = static_cast<int>((end - start) / step_size) + 1;
        t = Eigen::VectorXd::LinSpaced(num_points, start, end); // column vector (num_points,1)

        Eigen::MatrixXd cos_omega_t = (omega * t).array().cos();
        Eigen::MatrixXd sin_omega_t = (omega * t).array().sin();

        Eigen::MatrixXd x2 = R * cos_omega_t;
        Eigen::MatrixXd y2 = R * sin_omega_t;

        Eigen::MatrixXd Vx2 = -omega * R * sin_omega_t;
        Eigen::MatrixXd Vy2 = omega * R * cos_omega_t;

        Eigen::MatrixXd Ax2 = -1.0 * omega * omega * R * cos_omega_t;
        Eigen::MatrixXd Ay2 = -1.0 * omega * omega * R * sin_omega_t;

        std::vector<double> sig_a = scenario_params_.sig_a;
        int seed = scenario_params_.seed;
        add_noise(Ax1, sig_a, seed);
        seed += 1;
        add_noise(Ay1, sig_a, seed);
        seed += 1;
        add_noise(Ax2, sig_a, seed);
        seed += 1;
        add_noise(Ay2, sig_a, seed);

        double dt = kf_params_.dt;

        x1 = x1.array() + 0.5 * Ax1.array() * dt * dt;
        y1 = y1.array() + 0.5 * Ay1.array() * dt * dt;
        x2 = x2.array() + 0.5 * Ax2.array() * dt * dt;
        y2 = y2.array() + 0.5 * Ay2.array() * dt * dt;

        Vx1 = Vx1.array() + Ax1.array() * dt;
        Vy1 = Vy1.array() + Ay1.array() * dt;

        Vx2 = Vx2.array() + Ax2.array() * dt;
        Vy2 = Vy2.array() + Ay2.array() * dt;

        // std::cout << "x1 shape: (" << x1.rows() << ", " << x1.cols() << ")\n";
        // std::cout << "y1 shape: (" << y1.rows() << ", " << y1.cols() << ")\n";
        // std::cout << "x2 shape: (" << x2.rows() << ", " << x2.cols() << ")\n";
        // std::cout << "y2 shape: (" << y2.rows() << ", " << y2.cols() << ")\n";
        // std::cout << "Vx1 shape: (" << Vx1.rows() << ", " << Vx1.cols() << ")\n";
        // std::cout << "Vy1 shape: (" << Vy1.rows() << ", " << Vy1.cols() << ")\n";
        // std::cout << "Vx2 shape: (" << Vx2.rows() << ", " << Vx2.cols() << ")\n";
        // std::cout << "Vy2 shape: (" << Vy2.rows() << ", " << Vy2.cols() << ")\n";

        // Stack trajectories
        Eigen::MatrixXd x(x1.rows()+x2.rows(), x1.cols());
        x.topLeftCorner(x1.rows(), x1.cols()) = x1;
        x.bottomLeftCorner(x2.rows(), x2.cols()) = x2;

        Eigen::MatrixXd y(y1.rows()+y2.rows(), y1.cols());
        y.topLeftCorner(y1.rows(), y1.cols()) = y1;
        y.bottomLeftCorner(y2.rows(), y2.cols()) = y2;

        Eigen::MatrixXd vx(Vx1.rows()+Vx2.rows(), Vx1.cols());
        vx.topLeftCorner(Vx1.rows(), Vx1.cols()) = Vx1;
        vx.bottomLeftCorner(Vx2.rows(), Vx2.cols()) = Vx2;

        Eigen::MatrixXd vy(Vy1.rows()+Vy2.rows(), Vy1.cols());
        vy.topLeftCorner(Vy1.rows(), Vy1.cols()) = Vy1;
        vy.bottomLeftCorner(Vy2.rows(), Vy2.cols()) = Vy2;

        Eigen::MatrixXd ax(Ax1.rows()+Ax2.rows(), Ax1.cols());
        ax.topLeftCorner(Ax1.rows(), Ax1.cols()) = Ax1;
        ax.bottomLeftCorner(Ax2.rows(), Ax2.cols()) = Ax2;

        Eigen::MatrixXd ay(Ay1.rows()+Ay2.rows(), Ay1.cols());
        ay.topLeftCorner(Ay1.rows(), Ay1.cols()) = Ay1;
        ay.bottomLeftCorner(Ay2.rows(), Ay2.cols()) = Ay2;

        Eigen::MatrixXd X(x.rows(), 6);
        X.col(0) = x;
        X.col(1) = y;
        X.col(2) = vx;
        X.col(3) = vy;
        X.col(4) = ax;
        X.col(5) = ay;

        gtData.ax = ax;
        gtData.ay = ay;
        gtData.vx = vx;
        gtData.vy = vy;
        gtData.x = x;
        gtData.y = y;

        gtData.R = (x.array().pow(2) + y.array().pow(2)).array().sqrt();
        gtData.phi = (y.array() / x.array()).atan();

        gtData.X = X;
    }

    void GetMeasData(const GtData& gtData, MeasData& measData) const {
        Eigen::VectorXd R = gtData.R;
        Eigen::VectorXd phi = gtData.phi;

        Eigen::MatrixXd Z(R.rows(), R.cols() + phi.cols());
        Z.col(0) = R;
        Z.col(1) = phi;

        add_noise(Z, kf_params_.r_m, noise_gen_params_.seed);

        measData.R = Z.col(0);
        measData.phi = Z.col(1);
    }

    GtData GetGtDataTemplate() const {
        GtData gtData;
        return gtData;
    }

    MeasData GetMeasDataTemplate() const {
        MeasData measData;
        return measData;
    }

    static Scenario2& GetInstance() {
        static Scenario2 scenario2;
        return scenario2;
    }

public:
    KF kf_params_;
    NoiseGen noise_gen_params_;
    Scenario scenario_params_;
};

#endif // SCENARIOS_H_