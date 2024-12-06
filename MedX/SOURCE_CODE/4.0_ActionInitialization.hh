#ifndef ActionInitialization_hh
#define ActionInitialization_hh

#include "G4VUserActionInitialization.hh"

#include "3.0_DetectorConstruction.hh"
#include "5.0_PrimaryGenerator.hh"
#include "6.0_RunAction.hh"
#include "6.1_Run.hh"
#include "8.0_SteppingAction.hh"

class ActionInitialization : public G4VUserActionInitialization
{
    public:
        ActionInitialization(DetectorConstruction * detector);
        ~ActionInitialization();

        virtual void Build() const;
        virtual void BuildForMaster() const;

        DetectorConstruction * fDetector; 
};

#endif