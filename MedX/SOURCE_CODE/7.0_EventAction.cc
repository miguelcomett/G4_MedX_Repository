#include "7.0_EventAction.hh"

EventAction::EventAction(RunAction * runAction)
{
    fRunAction = runAction;
    fEDep = 0.0;
}

EventAction::~EventAction(){}

void EventAction::BeginOfEventAction(const G4Event * event) {fEDep = 0.0;}
void EventAction::EndOfEventAction(const G4Event * event) 
{ 
    totalEvents = G4RunManager::GetRunManager() -> GetNumberOfEventsToBeProcessed();
    eventID = event -> GetEventID();

    if (eventID == std::ceil(totalEvents*.25)) 
    { 
        nowTime1 = std::chrono::system_clock::now();
        nowTime2 = std::chrono::system_clock::to_time_t(nowTime1);
        nowTime3 = std::localtime(&nowTime2);
        std::cout << "\033[32;1m=== Progress: 25% ====== Time: " 
        << std::put_time(nowTime3, "%H:%M:%S") << "\033[0m" << std::endl;
    }

    if (eventID == std::ceil(totalEvents*.50)) 
    { 
        nowTime1 = std::chrono::system_clock::now();
        nowTime2 = std::chrono::system_clock::to_time_t(nowTime1);
        nowTime3 = std::localtime(&nowTime2);
        std::cout << "\033[32;1m==== Progress: 50% ====== Time: "
        << std::put_time(nowTime3, "%H:%M:%S") << "\033[0m" << std::endl;
    }

    if (eventID == std::ceil(totalEvents*.75)) 
    { 
        nowTime1 = std::chrono::system_clock::now();
        nowTime2 = std::chrono::system_clock::to_time_t(nowTime1);
        nowTime3 = std::localtime(&nowTime2);
        std::cout << "\033[32;1m===== Progress: 75% ====== Time: "
        << std::put_time(nowTime3, "%H:%M:%S") << "\033[0m" << std::endl;
    }

    // if (arguments == 1 || arguments == 2)
    // {
    //     G4AnalysisManager * analysisManager = G4AnalysisManager::Instance();
    //     if (fEDep > 0.0) 
    //     {
    //         analysisManager -> FillNtupleDColumn(4, 0, fEDep);
    //         analysisManager -> AddNtupleRow(4);
    //     }
    // }

    fRunAction -> AddEdep(fEDep);
}