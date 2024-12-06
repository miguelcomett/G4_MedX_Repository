#include "4.0_ActionInitialization.hh"

ActionInitialization::ActionInitialization(DetectorConstruction * detector) : G4VUserActionInitialization(), fDetector(detector) {}
ActionInitialization::~ActionInitialization(){}

void ActionInitialization::BuildForMaster() const 
{
    RunAction * runAction = new RunAction();
    SetUserAction(runAction);
}

void ActionInitialization::Build() const
{
    PrimaryGenerator * generator = new PrimaryGenerator(fDetector);
    SetUserAction(generator);
    
    RunAction * runAction = new RunAction();
    SetUserAction(runAction);

    EventAction * eventAction = new EventAction(runAction);
    SetUserAction(eventAction);

    SteppingAction * steppingAction = new SteppingAction(eventAction);
    SetUserAction(steppingAction);
}
