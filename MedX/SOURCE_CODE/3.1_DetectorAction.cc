#include "3.1_DetectorAction.hh"

SensitiveDetector::SensitiveDetector(G4String name):G4VSensitiveDetector(name){}
SensitiveDetector::~SensitiveDetector(){}

G4bool SensitiveDetector::ProcessHits(G4Step * currentStep, G4TouchableHistory * ROhist)
{
    particleTrack = currentStep -> GetTrack();
    particleTrack -> SetTrackStatus(fStopAndKill);

    particleName = particleTrack -> GetDefinition() -> GetParticleName();
    energyKeV = currentStep -> GetPreStepPoint() -> GetKineticEnergy() / keV;

    if (particleName == "gamma" && energyKeV >= 1.0)
    {
        preStepPoint  = currentStep -> GetPreStepPoint();
        postStepPoint = currentStep -> GetPostStepPoint();
        
        posPhoton = preStepPoint -> GetPosition();
        momPhoton = preStepPoint -> GetMomentum();

        Energy = preStepPoint -> GetKineticEnergy() / keV;
        Wavelength = (1.239841939 * eV / momPhoton.mag()) * 1E+03;
        
        touchable = currentStep -> GetPreStepPoint() -> GetTouchable();
        copyNo = touchable -> GetCopyNumber();
        detectorVolume = touchable -> GetVolume();
        posDetector = detectorVolume -> GetTranslation();
    
        Event = G4RunManager::GetRunManager() -> GetCurrentEvent() -> GetEventID();
        analysisManager = G4AnalysisManager::Instance();
        
        if (arguments == 1 || arguments == 2)
        {
            analysisManager -> FillNtupleIColumn(0, 0, Event);
            analysisManager -> FillNtupleDColumn(0, 1, posPhoton[0]);
            analysisManager -> FillNtupleDColumn(0, 2, posPhoton[1]);
            analysisManager -> FillNtupleDColumn(0, 3, posPhoton[2]);
            if (Wavelength > 0.0) {analysisManager -> FillNtupleDColumn(0, 4, Wavelength);}
            analysisManager -> AddNtupleRow(0);
            
            analysisManager -> FillNtupleIColumn(1, 0, Event);
            analysisManager -> FillNtupleDColumn(1, 1, posDetector[0]);
            analysisManager -> FillNtupleDColumn(1, 2, posDetector[1]);
            analysisManager -> FillNtupleDColumn(1, 3, posDetector[2]);
            analysisManager -> AddNtupleRow(1);
        }

        if (arguments == 4)
        {
            analysisManager -> FillNtupleDColumn(0, 0, Energy);
            analysisManager -> AddNtupleRow(0);
        }

        if (arguments == 5)
        {
            Decimals = 2;
            scaleFactor = std::pow(10, Decimals);

            Xpos = std::round(posPhoton[0] * scaleFactor) / scaleFactor;
            Ypos = std::round(posPhoton[1] * scaleFactor) / scaleFactor;

            const DetectorConstruction * detectorConstruction = static_cast<const DetectorConstruction*>(G4RunManager::GetRunManager() -> GetUserDetectorConstruction());
            is3DModel = detectorConstruction -> Getis3DModel();

            if ( (is3DModel == true) && (posPhoton[0]<230*mm && posPhoton[0]>-230*mm && posPhoton[1]<240*mm && posPhoton[1]>-240*mm) )
            { 
                analysisManager -> FillNtupleFColumn(0, 0, Xpos);
                analysisManager -> FillNtupleFColumn(0, 1, Ypos);
                analysisManager -> AddNtupleRow(0);
            }
            
            if (is3DModel == false)
            {
                analysisManager -> FillNtupleFColumn(0, 0, Xpos);
                analysisManager -> FillNtupleFColumn(0, 1, Ypos);
                analysisManager -> AddNtupleRow(0);
            }
        }
    }

    return true;
}