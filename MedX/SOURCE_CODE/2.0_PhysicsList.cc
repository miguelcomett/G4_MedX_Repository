#include "2.0_PhysicsList.hh"

PhysicsList::PhysicsList()
{
    if (arguments == 1 || arguments == 2 || arguments == 5) {RegisterPhysics(new G4EmStandardPhysics(0));}
    if (arguments == 3) {RegisterPhysics(new G4EmStandardPhysics_option1(0));}
    if (arguments == 4) {RegisterPhysics(new G4EmStandardPhysics_option4(0));}
    
    RegisterPhysics(new G4OpticalPhysics(0));
}

PhysicsList::~PhysicsList(){}

void PhysicsList::ConstructParticle() 
{
    G4VModularPhysicsList::ConstructParticle();
}

void PhysicsList::ConstructProcess() 
{
    G4VModularPhysicsList::ConstructProcess();
}