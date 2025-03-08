#include "3.1_DetectorAction.hh"

SensitiveDetector::SensitiveDetector(G4String name):G4VSensitiveDetector(name){}
SensitiveDetector::~SensitiveDetector(){}

G4bool SensitiveDetector::ProcessHits(G4Step * currentStep, G4TouchableHistory * ROhist)
{
    particleTrack = currentStep -> GetTrack();
    particleTrack -> SetTrackStatus(fStopAndKill);

    preStepPoint  = currentStep -> GetPreStepPoint();
    
    particleName = particleTrack -> GetDefinition() -> GetParticleName();
    photonEnergy = preStepPoint -> GetKineticEnergy();
    photonEnergy = photonEnergy / keV;

    if (particleName == "gamma" && photonEnergy >= 1.0)
    {   
        posPhoton = preStepPoint -> GetPosition();

        Wavelength = (1.239841939 / photonEnergy);
        
        touchable = currentStep -> GetPreStepPoint() -> GetTouchable();
        detectorVolume = touchable -> GetVolume();
        posDetector = detectorVolume -> GetTranslation();
    
        analysisManager = G4AnalysisManager::Instance();
        
        if (arguments == 1 || arguments == 2)
        {
            analysisManager -> FillNtupleDColumn(0, 0, posPhoton[0]);
            analysisManager -> FillNtupleDColumn(0, 1, posPhoton[1]);
            analysisManager -> FillNtupleDColumn(0, 2, posDetector[0]);
            analysisManager -> FillNtupleDColumn(0, 3, posDetector[1]);
            analysisManager -> AddNtupleRow(0);
            
            analysisManager -> FillNtupleDColumn(4, 0, photonEnergy);
            analysisManager -> FillNtupleDColumn(4, 1, Wavelength);
            analysisManager -> AddNtupleRow(4);
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

            const DetectorConstruction * detectorConstruction 
            = static_cast<const DetectorConstruction*>(G4RunManager::GetRunManager() -> GetUserDetectorConstruction());
            if (detectorConstruction) {is3DModel = detectorConstruction -> Getis3DModel();}
            if (detectorConstruction) {thoraxAngle = detectorConstruction -> GetThoraxAngle();} else {thoraxAngle = 0;}

            gunAngle = thoraxAngle * (2*pi / 360);
            model_width = 250; 
            model_depth = 150;
            minimum_span = model_depth / model_width;
            minimum_span = minimum_span * 1.15; // padding for safety
            x_lim = model_width * ( (std::cos(gunAngle) * std::cos(gunAngle)) * (1-minimum_span) + minimum_span);
            x_lim = x_lim * mm;

            y_lim = 250 * mm;

            // if ( (is3DModel == true) && (posPhoton[0]<230*mm && posPhoton[0]>-230*mm && posPhoton[1]<240*mm && posPhoton[1]>-240*mm) )
            if ( (is3DModel == true) && (posPhoton[0]<x_lim && posPhoton[0]>-x_lim && posPhoton[1]<y_lim && posPhoton[1]>-y_lim) )
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


// postStepPoint = currentStep -> GetPostStepPoint();
// momPhoton = preStepPoint -> GetMomentum();
// Energy = preStepPoint -> GetKineticEnergy() / keV;
// copyNo = touchable -> GetCopyNumber();
// Event = G4RunManager::GetRunManager() -> GetCurrentEvent() -> GetEventID();