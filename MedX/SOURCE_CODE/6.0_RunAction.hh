#ifndef RunAction_hh
#define RunAction_hh

#include <iomanip>
#include <ctime> 
#include <chrono>
#include <iostream>
#include <vector> 
#include <filesystem>
#include <string>
#include <regex>
#include <thread>

#include <TFile.h>
#include <TTree.h>
#include "Randomize.hh"
#include <G4RunManager.hh>
#include <G4AccumulableManager.hh>
#include "G4UIManager.hh"
#include "G4AnalysisManager.hh"
#include "G4Run.hh"
#include "G4Threading.hh"

#include "G4UserRunAction.hh"
#include "3.0_DetectorConstruction.hh"
#include "5.0_PrimaryGenerator.hh"
#include "6.1_Run.hh"

extern int arguments;

class EventAction;
class SteppingAction;

class RunAction : public G4UserRunAction
{
    public:

        RunAction();
        ~RunAction(); 

        void BeginOfRunAction(const G4Run * thisRun) override;
        void EndOfRunAction  (const G4Run * thisRun) override;

        G4Run * GenerateRun() override;

        void AddEDep(G4double EDepEvent) {EDepSum += EDepEvent;}
        void MergeEnergySpectra();
        void MergeRootFiles(const std::string & fileName, const std::string & tempDirectory, const std::string & rootDirectory);

    private:

        G4AnalysisManager * analysisManager;
        G4AccumulableManager * accumulableManager;

        Run * customRun;
        const Run * currentRun;
        const PrimaryGenerator * primaryGenerator = static_cast <const PrimaryGenerator*> (G4RunManager::GetRunManager() -> GetUserPrimaryGeneratorAction());
        const DetectorConstruction * detectorConstruction = static_cast <const DetectorConstruction*> (G4RunManager::GetRunManager() -> GetUserDetectorConstruction());  

        G4Accumulable <G4double> EDepSum = 0.0;
        std::vector <G4LogicalVolume*> scoringVolumes;

        std::map<G4String, G4double> energyDepositionMap;
        std::map<G4float, G4int> energyHistogram;

        std::chrono::system_clock::time_point simulationStartTime, simulationEndTime;
        std::time_t now_start;
        std::tm * now_tm_0;
        std::time_t now_end;
        std::tm * now_tm_1;

        G4ParticleDefinition * particle;

        std::string currentPath, tempDirectory, rootDirectory, mergedFileName;
        G4String particleName, baseName, fileName, haddCommand, tissueName;
        G4int numberOfEvents, runID, totalNumberOfEvents, threadID, GunMode, frequency, fileIndex;
        G4float primaryEnergy, energies;
        G4double energy, sampleMass, totalMass, durationInSeconds, TotalEnergyDeposit, radiationDose, tissueEDep;

        const G4double milligray = 1.0e-3*gray;
        const G4double microgray = 1.0e-6*gray;
        const G4double nanogray  = 1.0e-9*gray;
        const G4double picogray  = 1.0e-12*gray;
};

#endif