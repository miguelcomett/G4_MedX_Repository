#ifndef PHYSICSLISTS_HH
#define PHYSICSLISTS_HH

#include "G4StepLimiterPhysics.hh"

#include "G4VModularPhysicsList.hh"
#include "G4OpticalPhysics.hh"

#include "G4EmLivermorePhysics.hh"
#include "G4EmStandardPhysics_option1.hh"
#include "G4EmStandardPhysics_option3.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmStandardPhysics.hh"

extern int arguments; 

class PhysicsList : public G4VModularPhysicsList
{
    public:
        PhysicsList();
        ~PhysicsList();   
};

#endif


// in detector construction:
// stepLimit = new G4UserLimits(1e-3 * mm);
// logicHeart -> SetUserLimits(stepLimit);