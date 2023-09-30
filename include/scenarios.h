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
        
        Eigen::VectorXd sig_a_vec(sig_a.size());
        for (int i = 0; i < sig_a.size(); i++) {
            sig_a_vec[i] = sig_a[i];
        }

        double start = 0;
        double end = (double)(dt_*N);
        double step_size = dt_;

        // int num_points = static_cast<int>((end - start) / step_size) + 1;
        Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(N, start, end); // column vector (N,1)

        Eigen::MatrixXd a_vec(t.rows(), 1);
        a_vec.setConstant(a);

        add_noise(a_vec, sig_a_vec);

        Eigen::VectorXd x = Eigen::VectorXd(t.rows()).setConstant(x0).array() + t.array() * v0 + 0.5 * a_vec.array() * t.array() * t.array();
        Eigen::VectorXd v = Eigen::VectorXd(t.rows()).setConstant(v0).array() + t.array() * a_vec.array();
        a_vec = a_vec.array() - g;

        gt_data.alt = x;
        gt_data.vel = v;
        gt_data.acc = a_vec;
    }

    void GetMeasData(const GtData& gt_data, MeasData& meas_data) const {
        auto r = kf_params_.r;
        Eigen::VectorXd r_vec(r.size());
        for (int i = 0; i < r.size(); i++) {
            r_vec[i] = r[i];
        }

        std::vector<double> sig_a = scenario_params_.sig_a;
        Eigen::VectorXd sig_a_vec(sig_a.size());
        for (int i = 0; i < sig_a.size(); i++) {
            sig_a_vec[i] = sig_a[i];
        }


        int alt_noise_params = alt_noise_gen_params_.seed;
        auto true_alt = gt_data.alt;
        auto Z = Eigen::MatrixXd(true_alt);
        add_noise(Z, r_vec, alt_noise_params);

        meas_data.alt = Eigen::VectorXd(Z.col(0));
    }

    void GetInpData(const GtData& gt_data, InpData& inp_data) const {
        std::vector<double> sig_a = scenario_params_.sig_a;
        Eigen::VectorXd sig_a_vec(sig_a.size());
        for (int i = 0; i < sig_a.size(); i++) {
            sig_a_vec[i] = sig_a[i];
        }
        
        int acc_noise_params = acc_noise_gen_params_.seed;
        auto true_acc = gt_data.acc;
        auto A = Eigen::MatrixXd(true_acc);
        add_noise(A, sig_a_vec, acc_noise_params);

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

    Eigen::MatrixXd GetProcessNoiseCovariance(int state_size) {
        const double dt = dt_;
        const double sig_a = scenario_params_.sig_a[0];
        const double var_acc = sig_a * sig_a;
        Eigen::MatrixXd process_noise_covariance = Eigen::MatrixXd(state_size, state_size);
        process_noise_covariance << pow(dt,4)*var_acc/4.0, pow(dt,3)*var_acc/2.0, pow(dt,3)*var_acc/2.0, pow(dt,2)*var_acc;

        return process_noise_covariance;
    }

    Eigen::MatrixXd GetMeasurementCovariance(int measurement_size) {
        const double sigma_meas = kf_params_.r[0];
        Eigen::MatrixXd measurement_covariance = Eigen::MatrixXd(measurement_size, measurement_size);
        measurement_covariance << sigma_meas * sigma_meas;

        return measurement_covariance;
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

template <typename T>
T& get_scenario (int scenario_num) {
    switch (scenario_num) {
        case 1:
            return Scenario1::GetInstance();
            break;
        
        default:
            std::cout << "There is no such scenario. Returning Scenario1 as a default." << std::endl;
            return Scenario1::GetInstance();
            break;
    }
}


#endif // SCENARIOS_H_