#include <iostream>
#include <vector>
#include "scenarios.h"

#include "lkf.h"
#include "helper_functions.h"

int main()
{
    Scenario2& scenario2 = Scenario2::GetInstance();

    auto gtData = scenario2.GetGtDataTemplate();
    scenario2.GetGtData(gtData);

    auto measData = scenario2.GetMeasDataTemplate();
    scenario2.GetMeasData(gtData, measData);

    return 0;
}