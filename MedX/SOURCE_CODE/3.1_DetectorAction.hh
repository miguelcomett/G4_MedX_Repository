#ifndef DetectorAction_hh
#define DetectorAction_hh

#include "G4VSensitiveDetector.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

#include "3.0_DetectorConstruction.hh"

extern int arguments;

class SensitiveDetector:public G4VSensitiveDetector
{
    public:

        SensitiveDetector(G4String);
        ~SensitiveDetector();
    
        virtual G4bool ProcessHits(G4Step *, G4TouchableHistory *);
        
    private: 
    
        G4AnalysisManager * analysisManager;
        G4VPhysicalVolume * detectorVolume;
        const G4VTouchable * touchable;
        G4StepPoint * preStepPoint, * postStepPoint;
        G4Track * particleTrack;

        G4bool is3DModel;
        const G4double pi = 3.14159265358979323846;
        G4String particleName;
        G4int digits, defaultDecimals, copyNo, Event, Decimals, scaleFactor;
        G4float Xpos, Ypos;
        G4double Wavelength, photonEnergy, photonEnergy_keV, thoraxAngle, gunAngle, model_width, model_depth, minimum_span, x_lim, y_lim, 
                             DetectorEnergyDeposition;
        
        G4ThreeVector posPhoton, posDetector;

        const G4UserRunAction * userRunAction = G4RunManager::GetRunManager() -> GetUserRunAction();
};

#endif