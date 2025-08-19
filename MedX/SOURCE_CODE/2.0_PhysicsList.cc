#include "2.0_PhysicsList.hh"

PhysicsList::PhysicsList()
{
    if (arguments == 1 || arguments == 2 || arguments == 3) {RegisterPhysics(new G4EmStandardPhysics(0));}
    if (arguments == 3) {RegisterPhysics(new G4EmStandardPhysics_option1(0));}
    if (arguments == 4) 
    {
        RegisterPhysics(new G4EmLivermorePhysics(0));

        // Atomic de-excitation
        auto EM_Parameters = G4EmParameters::Instance();
        EM_Parameters -> SetFluo(true);     
        EM_Parameters -> SetAuger(true);    
        EM_Parameters -> SetPixe(true);     
        EM_Parameters -> SetDeexcitationIgnoreCut(true); 
        EM_Parameters -> SetMinEnergy(100*CLHEP::eV); 
        EM_Parameters -> SetMaxEnergy(1*CLHEP::GeV); 
    }
    
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

void PhysicsList::SetCuts() 
{
    // Define very low production cuts for low-energy X-ray simulation
    SetDefaultCutValue(0.1*CLHEP::um); 
    SetCutValue(0.1*CLHEP::um, "gamma");
    SetCutValue(0.05*CLHEP::um, "e-");   
    SetCutValue(0.05*CLHEP::um, "e+");   

    G4VUserPhysicsList::SetCuts();  
}