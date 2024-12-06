#ifndef SteppingAction_hh
#define SteppingAction_hh

#include <algorithm>

#include "G4UserSteppingAction.hh"
#include "G4Step.hh"
#include "G4RunManager.hh"

#include "3.0_DetectorConstruction.hh"
#include "6.1_Run.hh"
#include "7.0_EventAction.hh"

extern int arguments;

class SteppingAction : public G4UserSteppingAction
{
    public:

        SteppingAction(EventAction * eventAction);
        ~ SteppingAction();

        virtual void UserSteppingAction(const G4Step *);
    
    private:

        G4String processName;
        G4double threshold;
        G4int trackID;
        G4double EDep;
        G4double worldMaxX, worldMinX, worldMaxY, worldMinY, worldMaxZ, worldMinZ;

        G4ThreeVector position, currentPosition;
        G4Track * track;

        EventAction * fEventAction;
        G4VPhysicalVolume * currentVolume;
        G4LogicalVolume * scoringVolume, * Volume, * currentLogicVolume;
        G4StepPoint * endPoint;
        const DetectorConstruction * detectorConstruction;

        struct ParticleData {G4ThreeVector lastPosition; int stuckStepCount;};
        std::map<G4int, ParticleData> stuckParticles;
};

#endif