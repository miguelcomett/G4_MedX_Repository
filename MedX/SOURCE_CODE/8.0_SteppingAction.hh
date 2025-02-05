#ifndef SteppingAction_hh
#define SteppingAction_hh

#include <algorithm>

#include "G4Step.hh"
#include "G4RunManager.hh"

#include "G4UserSteppingAction.hh"
#include "3.0_DetectorConstruction.hh"
#include "6.1_Run.hh"
#include "7.0_EventAction.hh"

extern int arguments;

class SteppingAction : public G4UserSteppingAction
{
    public:

        SteppingAction();
        ~ SteppingAction();

        virtual void UserSteppingAction(const G4Step *);

        std::map<G4String, G4double> GetEnergyDepositionMap() const {return energyDepositionMap;}
    
    private:

        G4String processName, volumeName;
        G4bool Stuck;
        G4int trackID;
        G4double worldMaxX, worldMinX, worldMaxY, worldMinY, worldMaxZ, worldMinZ, energyDeposition, threshold;

        G4ThreeVector position, currentPosition;

        std::vector<G4LogicalVolume*> scoringVolumes;
        std::map<G4String, G4double> energyDepositionMap;

        struct ParticleData {G4ThreeVector lastPosition; int stuckStepCount;};
        std::map<G4int, ParticleData> stuckParticles;

        G4VPhysicalVolume * currentVolume;
        G4LogicalVolume * scoringVolume, * Volume, * currentLogicVolume;
        G4StepPoint * endPoint;
        
        G4Track * track;
        Run * run;

        const G4VUserDetectorConstruction * userDetectorConstruction = G4RunManager::GetRunManager() -> GetUserDetectorConstruction();
        const DetectorConstruction * detectorConstruction = static_cast <const DetectorConstruction*> (userDetectorConstruction);

        const G4UserRunAction * userRunAction = G4RunManager::GetRunManager() -> GetUserRunAction();
        const RunAction * constRunAction = dynamic_cast <const RunAction*> (userRunAction);
        RunAction * runAction = const_cast <RunAction*> (constRunAction);
};

#endif