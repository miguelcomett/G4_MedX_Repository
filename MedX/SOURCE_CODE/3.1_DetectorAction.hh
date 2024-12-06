#ifndef DetectorAction_hh
#define DetectorAction_hh

#include "G4VSensitiveDetector.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

#include "3.0_DetectorConstruction.hh"

extern int arguments;

class SensitiveDetector : public G4VSensitiveDetector
{
    public:

        SensitiveDetector(G4String);
        ~SensitiveDetector();
    
    private: 

        virtual G4bool ProcessHits(G4Step *, G4TouchableHistory *);
        G4int digits, defaultDecimals, copyNo, Event;
        G4double Wavelength, Energy;
};

#endif