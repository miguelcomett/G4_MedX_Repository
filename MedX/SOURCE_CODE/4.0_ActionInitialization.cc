#include "4.0_ActionInitialization.hh"

ActionInitialization::ActionInitialization(){}
ActionInitialization::~ActionInitialization()
{
    delete myPrimaryGenerator;
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
    PrimaryGenerator * myPrimaryGenerator = new PrimaryGenerator();
    SetUserAction(myPrimaryGenerator);
    
    RunAction * myRunAction = new RunAction();
    SetUserAction(myRunAction);
    
    EventAction * myEventAction = new EventAction();
    SetUserAction(myEventAction);
    
    SteppingAction * mySteppingAction = new SteppingAction();
    SetUserAction(mySteppingAction);
}