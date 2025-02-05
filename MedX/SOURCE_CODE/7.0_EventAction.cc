#include "7.0_EventAction.hh"

EventAction::EventAction(){}
EventAction::~EventAction(){}

void EventAction::BeginOfEventAction(const G4Event * event) {EDepEvent = 0.0;}
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
}