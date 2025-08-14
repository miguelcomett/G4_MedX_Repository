#ifndef PrimaryGenerator_hh
#define PrimaryGenerator_hh

#include <iomanip>
#include <vector>
#include <fstream>
#include <ctime>
#include <map>
#include <cmath>

#include "Randomize.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4RunManager.hh"
#include "G4VUserPrimaryGeneratorAction.hh"

#include "3.0_DetectorConstruction.hh"
#include "6.1_Run.hh"
#include "5.1_GeneratorMessenger.hh"

class PrimaryGeneratorMessenger;

class PrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
    public:

        PrimaryGenerator();
        ~PrimaryGenerator();

        virtual void GeneratePrimaries(G4Event *);
        void SetGunXpos(G4double newXpos);
        void SetGunYpos(G4double newYpos);
        void SetGunZpos(G4double newZpos);
        void SetGunXcos(G4bool newXtriangular);
        void SetGunXGauss(G4bool newXgauss);
        void SetGunSpanX(G4double newSpanX);
        void SetGunSpanY(G4double newSpanY);
        void SetGunAngle(G4double newAngle); 
        void SetGunMode(G4int newMode); 
	
        G4ParticleGun * GetParticleGun() const {return particleGun;}

        G4int GetGunMode() const {return SpectraMode;}
        std::map <G4float, G4int> GetEnergySpectra() const {return energyHistogram;}
        void ReadSpectrumFromFile(const std::string & filename, std::vector<G4double> & xx, std::vector<G4double> & yy, G4int & energyDataPoints);
        G4double InverseCumul();
        void SpectraFunction(); 
        G4double GetTotalEnergy() const {return totalEnergy;}; 
        
    private:

        std::map <G4float, G4int> energyHistogram; 

        G4ParticleGun * particleGun;    
        PrimaryGeneratorMessenger * GeneratorMessenger;

        const G4VUserDetectorConstruction * userDetectorConstruction = G4RunManager::GetRunManager() -> GetUserDetectorConstruction();
        const DetectorConstruction * detectorConstruction = static_cast <const DetectorConstruction*> (userDetectorConstruction);

        G4ParticleTable * particleTable;
        G4ParticleDefinition * particleName;

        G4String spectrumFile; 	       
        G4bool Xtriangular, newXtriangular, Xcos, Xgauss, newXgauss;
        G4int threadID, SpectraMode, Decimals, roundingScale, energyDataPoints, bins, sign;
        G4float RealEnergy;
        const G4double pi = 3.14159265358979323846;
        G4double x0, y0, z0, model_width, model_depth, minimum_span, thoraxAngle, gunAngle, Theta, Phi, AngleInCarts, Xpos, Ypos, Zpos, 
                SpanX, SpanY, GunAngle, random, peak, min, max,
                Y_max, X_random, Y_random, Alfa, Beta, Gamma, Delta, energy, intensity, totalEnergy;

        G4ThreeVector photonPosition, photonMomentum;
        std::vector<G4double> EnergyVector, IntensityVector, X_vector, Y_vector, Slopes_vector, Y_Cumulative;
};

#endif