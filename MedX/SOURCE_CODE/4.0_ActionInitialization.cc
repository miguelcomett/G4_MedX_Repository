#include "4.0_ActionInitialization.hh"

ActionInitialization::ActionInitialization(DetectorConstruction * detector)
{
    myDetector = detector;
}

ActionInitialization::~ActionInitialization()
{
    delete myGenerator;
    delete myRunAction;
    delete myEventAction;
    delete mySteppingAction;
}

void ActionInitialization::BuildForMaster() const
{
    RunAction * myRunAction = new RunAction();
    SetUserAction(myRunAction);
}

void ActionInitialization::Build() const
{
    PrimaryGenerator * myGenerator = new PrimaryGenerator(myDetector);
    SetUserAction(myGenerator);
    
    RunAction * myRunAction = new RunAction();
    SetUserAction(myRunAction);

    EventAction * myEventAction = new EventAction(myRunAction);
    SetUserAction(myEventAction);

    SteppingAction * mySteppingAction = new SteppingAction(myEventAction);
    SetUserAction(mySteppingAction);
}