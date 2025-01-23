#ifndef PrimaryGenerator_hh
#define PrimaryGenerator_hh

#include <iomanip>
#include <vector>
#include <fstream>
#include <ctime>

#include "Randomize.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4GeneralParticleSource.hh"
#include "G4PhysicalConstants.hh"
#include "G4RunManager.hh"

#include "3.0_DetectorConstruction.hh"
#include "6.1_Run.hh"
#include "5.1_GeneratorMessenger.hh"

class PrimaryGeneratorMessenger;
class DetectorConstruction;

class PrimaryGenerator:public G4VUserPrimaryGeneratorAction
{
    public:

        PrimaryGenerator(DetectorConstruction * detector);
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

        void ReadSpectrumFromFile(const std::string & filename, std::vector<G4double> & xx, std::vector<G4double> & yy, G4int & energyDataPoints);
        std::vector<G4double> GetEnergySpectra() const {return energySpectra;}
        G4double InverseCumul();
        
    private:

        G4ParticleGun * particleGun;        
        PrimaryGeneratorMessenger * GeneratorMessenger;
        DetectorConstruction * detectorConstruction;

        G4String particleName;
        G4ParticleTable * particleTable;
        G4ParticleDefinition * particle;

        const G4double pi = 3.14159265358979323846;
        G4int threadID, SpectraMode;
        G4bool Xtriangular, newXtriangular, Xcos, Xgauss, newXgauss;
        G4double x0, y0, z0, thoraxAngle, gunAngle, theta, phi, AngleInCarts, 
                 Xpos, Ypos, Zpos, SpanX, SpanY, GunAngle, RealEnergy, random, peak, min, max;

        G4ThreeVector photonPosition, photonMomentum;
        
        void SpectraFunction(); 
        
        std::vector<G4double> energySpectra;
        G4String spectrumFile; 	       
        G4int energyDataPoints = 0; //nb of points
        G4double fYmax;             //max(Y)
        std::vector<G4double> fX;   //abscisses X
        std::vector<G4double> fY;   //values of Y(X)
        std::vector<G4double> fSlp; //slopes
        std::vector<G4double> fYC;  //cumulative function of Y
};

#endif