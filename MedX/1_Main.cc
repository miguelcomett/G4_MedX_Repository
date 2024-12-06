#include <iostream>
#include <ctime> 
#include <chrono>
#include <iomanip>

#include "G4RunManager.hh"
#include "G4MTRunManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4UIManager.hh"
#include "G4UIExecutive.hh"
#include "G4VisManager.hh"
#include "G4VisExecutive.hh"
// #include "G4UIQt.hh"

#include "2.0_PhysicsList.hh"
#include "3.0_DetectorConstruction.hh"
#include "4.0_ActionInitialization.hh"

int arguments = 0;

int main(int argc, char** argv)
{
    arguments = (argc);

    #ifdef __APPLE__
        G4RunManager * runManager;
        if (argc == 1) {runManager = new G4RunManager();} 
        else {runManager = new G4MTRunManager();}
    #endif
    #ifdef _WIN32
        auto * runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
    #endif

    long seed = std::time(nullptr);
    CLHEP::HepRandom::setTheSeed(seed);

    runManager -> SetUserInitialization(new PhysicsList());
    DetectorConstruction * MyDetectorConstruction = new DetectorConstruction();
    runManager -> SetUserInitialization(MyDetectorConstruction);
    runManager -> SetUserInitialization(new ActionInitialization(MyDetectorConstruction));

    G4UImanager * UImanager = G4UImanager::GetUIpointer();
    
    if(argc == 1)
    {
        G4VisManager * visManager = new G4VisExecutive("quiet");  // "quiet", "all"
        visManager -> Initialize();

        //G4UIQt * UI = new G4UIQt(argc, argv); // Usando G4UIQt en lugar de G4UIExecutive
        G4UIExecutive * UI = nullptr;
        UI = new G4UIExecutive(argc, argv);
        UImanager -> ApplyCommand("/control/execute Visualization.mac");
        UI -> SessionStart();
        
        delete UI;
        delete visManager;
    }
    else
    {
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UImanager -> ApplyCommand(command + fileName);
    }

    delete runManager;
}