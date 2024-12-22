#include "8.0_SteppingAction.hh"
#include "G4UserLimits.hh"

SteppingAction::SteppingAction(EventAction * eventAction) {fEventAction = eventAction;}
SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step * step)
{
    track = step -> GetTrack();
    position = track -> GetPosition();
    worldMaxX =  500.1 * mm; worldMinX = -500.1 * mm; worldMaxY =  500.1 * mm; worldMinY = -500.1 * mm; worldMaxZ =  500.1 * mm; worldMinZ = -500.1 * mm;
    if (position.x() < worldMinX || position.x() > worldMaxX || position.y() < worldMinY || position.y() > worldMaxY || position.z() < worldMinZ || position.z() > worldMaxZ)  
    {track -> SetTrackStatus(fStopAndKill); G4cout << " ERROR: Particle outside world bounds!!!" << G4endl;}

    Volume = step -> GetPreStepPoint() -> GetTouchableHandle() -> GetVolume() -> GetLogicalVolume();
    detectorConstruction = static_cast < const DetectorConstruction *> (G4RunManager::GetRunManager() -> GetUserDetectorConstruction());
    scoringVolume = detectorConstruction -> GetScoringVolume();

    if (arguments == 1 || arguments == 2)
    {   
        std::vector<G4LogicalVolume*> scoringVolumes = detectorConstruction -> GetAllScoringVolumes();
        if (std::find(scoringVolumes.begin(), scoringVolumes.end(), Volume) == scoringVolumes.end()) {return;}
        {
            EDep = step -> GetTotalEnergyDeposit();
            if (EDep > 0.0) {fEventAction -> AddEDep(EDep);}
            fEventAction -> AddEDep(EDep);
            // G4cout << "Energy deposition (keV): " << EDep << G4endl; 
        }
    }

    if (arguments == 3) 
    {
        if(Volume != scoringVolume) { return; }
        // G4cout << "particle touches target: " << scoringVolume << G4endl;

        endPoint = step -> GetPostStepPoint();
        processName = endPoint -> GetProcessDefinedStep() -> GetProcessName();
        Run * run = static_cast <Run *> (G4RunManager::GetRunManager() -> GetNonConstCurrentRun()); 
        run -> CountProcesses(processName);

        G4RunManager::GetRunManager() -> AbortEvent();  // kill event after first interaction
    }

    if (arguments == 5)
    {   
        G4bool Stuck = true;
        
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

        std::vector<G4LogicalVolume*> scoringVolumes = detectorConstruction -> GetAllScoringVolumes();
        if (std::find(scoringVolumes.begin(), scoringVolumes.end(), Volume) == scoringVolumes.end()) {return;}
        {       
            EDep = step -> GetTotalEnergyDeposit();
            if (EDep > 0.0) {fEventAction -> AddEDep(EDep);}
            // G4cout << "Energy deposition (eV): " << G4BestUnit(EDep, "Energy") << G4endl; 
        }
    }
}