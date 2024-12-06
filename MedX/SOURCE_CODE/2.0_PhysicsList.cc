#include "2.0_PhysicsList.hh"

PhysicsList::PhysicsList()
{
    if (arguments != 5) {RegisterPhysics(new G4EmStandardPhysics(0));}
    if (arguments == 5) {RegisterPhysics(new G4EmStandardPhysics_option1(0));}
    
    RegisterPhysics(new G4OpticalPhysics(0));
}

PhysicsList::~PhysicsList(){}

// G4StepLimiterPhysics * stepLimitPhys = new G4StepLimiterPhysics();
// stepLimitPhys -> SetApplyToAll(true);
// RegisterPhysics(stepLimitPhys);