#ifndef EventAction_hh
#define EventAction_hh

#include <iostream>

#include "G4Event.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "G4Threading.hh"

#include "G4UserEventAction.hh"
#include "6.0_RunAction.hh"
#include "6.1_Run.hh"

extern int arguments;

class RunAction;

class EventAction : public G4UserEventAction
{
    public:
        
        EventAction(RunAction * runAction);
        ~EventAction();

        virtual void BeginOfEventAction(const G4Event *);
        virtual void EndOfEventAction  (const G4Event *);

        void AddEDepEvent(G4double EDepStep){EDepEvent = EDepEvent + EDepStep;};

    private:
        
        RunAction * myRunAction;
        
        G4int totalEvents, eventID;
        G4double EDepEvent;

        std::chrono::system_clock::time_point nowTime1;
        std::time_t nowTime2;
        std::tm * nowTime3;
};

#endif