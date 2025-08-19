#include "8.0_SteppingAction.hh"

SteppingAction::SteppingAction() {}
SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step * step)
{   
    Volume = step -> GetPreStepPoint() -> GetTouchableHandle() -> GetVolume() -> GetLogicalVolume();

    if (arguments == 1 || arguments == 2 || arguments == 3 || arguments == 4)
    {   
        scoringVolumes = detectorConstruction -> GetAllScoringVolumes();
        std::set <G4LogicalVolume*> scoringSet(scoringVolumes.begin(), scoringVolumes.end());
        
        if (scoringSet.count(Volume) > 0)
        {
            volumeName = Volume -> GetName();
            energyDeposition = step -> GetTotalEnergyDeposit();

            if (energyDeposition > 0.0) 
            {
                if (runAction) {runAction -> AddEDep(energyDeposition);}
                energyDepositionMap[volumeName] += energyDeposition;
            }
        }

        Stuck = false;
        
        if (Stuck == true) 
        {
            threshold = 1.0e-5; 
            currentVolume = step -> GetPreStepPoint() -> GetPhysicalVolume();
            trackID = track -> GetTrackID();
            currentPosition = track -> GetPosition();

            if (stuckParticles.find(trackID) != stuckParticles.end()) // Check if this track is already being monitored
            {
                ParticleData & data = stuckParticles[trackID];
                if ((currentPosition - data.lastPosition).mag() < threshold) 
                {
                    data.stuckStepCount++; // Increment stuck step count
                    if (data.stuckStepCount >= 5) 
                    {
                        track -> SetTrackStatus(fStopAndKill); 
                        std::cout << "• Killed stuck particle" << std::endl;
                        stuckParticles.erase(trackID);
                    }
                } 
                else {data.stuckStepCount = 0;}
                data.lastPosition = currentPosition;
            } 
            else {stuckParticles[trackID] = {currentPosition, 0};} // Add new track to monitoring
        }
    }

    if (arguments == 5) 
    {
        scoringVolume = detectorConstruction -> GetScoringVolume();

        if(Volume != scoringVolume) {return;}

        endPoint = step -> GetPostStepPoint();
        processName = endPoint -> GetProcessDefinedStep() -> GetProcessName();
        run = static_cast <Run*> (G4RunManager::GetRunManager() -> GetNonConstCurrentRun()); 
        run -> CountProcesses(processName);

        G4RunManager::GetRunManager() -> AbortEvent();
    }
}