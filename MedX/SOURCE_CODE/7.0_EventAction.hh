#ifndef EventAction_hh
#define EventAction_hh

#include <iostream>

#include "G4Event.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "G4Threading.hh"

#include "G4UserEventAction.hh"
#include "6.0_RunAction.hh"

extern int arguments;

class EventAction : public G4UserEventAction
{
    public:
        
        EventAction();
        ~EventAction();

        virtual void BeginOfEventAction(const G4Event *);
        virtual void EndOfEventAction  (const G4Event *);

    private:
        
        G4int totalEvents, eventID;
        G4double EDepEvent;

        std::chrono::system_clock::time_point nowTime1;
        std::time_t nowTime2;
        std::tm * nowTime3;
};

#endif