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
        G4String particleName;
        G4int digits, defaultDecimals, copyNo, Event, Decimals, scaleFactor;
        G4float Xpos, Ypos;
        G4double Wavelength, Energy, energyKeV;
        
        G4ThreeVector posPhoton, momPhoton, posDetector;
};

#endif