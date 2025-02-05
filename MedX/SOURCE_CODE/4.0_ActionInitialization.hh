#ifndef ActionInitialization_hh
#define ActionInitialization_hh

#include "G4VUserActionInitialization.hh"

#include "3.0_DetectorConstruction.hh"
#include "5.0_PrimaryGenerator.hh"
#include "6.0_RunAction.hh"
#include "6.1_Run.hh"
#include "8.0_SteppingAction.hh"

class ActionInitialization:public G4VUserActionInitialization
{
    public:

        ActionInitialization();
        ~ActionInitialization();

        virtual void BuildForMaster() const;
        virtual void Build() const;
    
        PrimaryGenerator * myPrimaryGenerator;
        RunAction * myRunAction;
        EventAction * myEventAction;
        SteppingAction * mySteppingAction;
};

#endif