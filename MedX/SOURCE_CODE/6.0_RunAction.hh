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

#include "Randomize.hh"
#include "G4UIManager.hh"
#include "G4UserRunAction.hh"
#include "G4AnalysisManager.hh"
#include "G4Run.hh"
#include <G4AccumulableManager.hh>
#include "G4Threading.hh"
#include "TFileMerger.h"
#include <TFile.h>
#include <TTree.h>
#include "G4Threading.hh"
#include "3.0_DetectorConstruction.hh"
#include "5.0_PrimaryGenerator.hh"
#include "6.1_Run.hh"

extern int arguments;

class RunAction : public G4UserRunAction
{
    public:

        RunAction();
        ~RunAction(); 

        void BeginOfRunAction(const G4Run * thisRun) override;
        void EndOfRunAction  (const G4Run * thisRun) override;

        G4Run * GenerateRun() override;

        void AddEdep (G4double edep);

        void MergeRootFiles();
        void RemoveJunkDataFromRoot(const std::string & mergedFilePath);

    private:

        Run * customRun = nullptr;

        const DetectorConstruction * detectorConstruction;

        const PrimaryGenerator * primaryGenerator;
        G4Accumulable <G4double> fEdep = 0.0;
        std::chrono::system_clock::time_point simulationStartTime, simulationEndTime;
        std::time_t now_end;
        std::tm * now_tm_1;

        G4ParticleDefinition * particle;

        G4String particleName, directory, fileName;
        G4int numberOfEvents, runID, index, totalNumberOfEvents, threadID;
        G4double energy, sampleMass, primaryEnergy, totalMass, durationInSeconds, TotalEnergyDeposit, radiationDose;

        const G4double milligray = 1.0e-3*gray;
        const G4double microgray = 1.0e-6*gray;
        const G4double nanogray  = 1.0e-9*gray;
        const G4double picogray  = 1.0e-12*gray;
};

#endif