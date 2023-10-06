#include <iostream>

#include "ukf.h"

int main() {
    UnscentedKalmanFilter ukf(6, 1, 1);

    ukf.CalculateWeights();

    return 0;
}