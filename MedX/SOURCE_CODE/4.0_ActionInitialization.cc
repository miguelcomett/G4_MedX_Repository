#include "4.0_ActionInitialization.hh"

ActionInitialization::ActionInitialization(DetectorConstruction * detector)
{
    fDetector = detector;
}
ActionInitialization::~ActionInitialization(){}

void ActionInitialization::BuildForMaster() const 
{
    RunAction * runAction = new RunAction();
    SetUserAction(runAction);
}

// ActionInitialization::~ActionInitialization()
// {
//     // Cleanup dynamically allocated memory
//     delete generator;
//     delete fRunAction;
//     delete fEventAction;
//     delete fSteppingAction;
// }

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