#include "6.0_RunAction.hh"

G4Mutex mergeMutex = G4MUTEX_INITIALIZER;
std::vector<G4float> masterEnergySpectra;

RunAction::RunAction()
{
    new G4UnitDefinition("milligray", "milliGy" , "Dose", milligray);
    new G4UnitDefinition("microgray", "microGy" , "Dose", microgray);
    new G4UnitDefinition("nanogray" , "nanoGy"  , "Dose", nanogray);
    new G4UnitDefinition("picogray" , "picoGy"  , "Dose", picogray);

    G4AccumulableManager * accumulableManager = G4AccumulableManager::Instance();
    accumulableManager -> RegisterAccumulable(fEdep);

    G4AnalysisManager * analysisManager = G4AnalysisManager::Instance();
    analysisManager -> SetDefaultFileType("root");
    analysisManager -> SetVerboseLevel(0);

    if (arguments == 1 || arguments == 2)
    {
        analysisManager -> CreateNtuple("Photons", "Photons");
        analysisManager -> CreateNtupleIColumn("Event_Count");
        analysisManager -> CreateNtupleDColumn("X_axis");
        analysisManager -> CreateNtupleDColumn("Y_axis");
        analysisManager -> CreateNtupleDColumn("Z_axis");
        analysisManager -> CreateNtupleDColumn("Photons'_Wavelengths_nm");
        analysisManager -> FinishNtuple(0);

        analysisManager -> CreateNtuple("Hits", "Hits");
        analysisManager -> CreateNtupleIColumn("Event_Count");
        analysisManager -> CreateNtupleDColumn("X_Detectors");
        analysisManager -> CreateNtupleDColumn("Y_Detectors");
        analysisManager -> CreateNtupleDColumn("Z_Detectors");
        analysisManager -> FinishNtuple(1);

        analysisManager -> CreateNtuple("Energy", "Energy");
        analysisManager -> CreateNtupleDColumn("Energy_Deposition_keV");
        analysisManager -> FinishNtuple(2);
    }

    if (arguments == 3)
    {
        analysisManager -> CreateNtuple("Transportation", "Transportation");
        analysisManager -> CreateNtupleDColumn("Mass_Attenuation");
        analysisManager -> CreateNtupleDColumn("Energy_keV");
        analysisManager -> CreateNtupleDColumn("Ratio");
        analysisManager -> FinishNtuple(0);
    }

    if (arguments == 4)
    {
        analysisManager -> CreateNtuple("Energy_Dist", "Energy_Dist");
        analysisManager -> CreateNtupleDColumn("Energies");
        analysisManager -> FinishNtuple(0);
    }

    if (arguments == 5)
    {
        analysisManager -> CreateNtuple("Hits", "Hits");
        analysisManager -> CreateNtupleFColumn("x_ax");
        analysisManager -> CreateNtupleFColumn("y_ax");
        analysisManager -> FinishNtuple(0);

        analysisManager -> CreateNtuple("Run Summary", "Run Summary");
        analysisManager -> CreateNtupleDColumn("Number_of_Photons");
        analysisManager -> CreateNtupleFColumn("Initial_Energy_keV", photonsEnergy);
        analysisManager -> CreateNtupleDColumn("Sample_Mass_kg");
        analysisManager -> CreateNtupleDColumn("EDep_Value_TeV");
        analysisManager -> CreateNtupleDColumn("Radiation_Dose_uSv");
        analysisManager -> FinishNtuple(1);
    }
}

RunAction::~RunAction(){}

G4Run * RunAction::GenerateRun() {customRun = new Run(); return customRun;}

void RunAction::AddEdep(G4double edep) {fEdep += edep;}

void RunAction::BeginOfRunAction(const G4Run * thisRun)
{
    threadID = G4Threading::G4GetThreadId();

    G4AccumulableManager * accumulableManager = G4AccumulableManager::Instance();
    accumulableManager -> Reset();

    std::string currentPath = std::filesystem::current_path().string(); // Obtener la ruta actual

    #ifdef __APPLE__
        std::string rootDirectory = std::filesystem::path(currentPath).string() + "/ROOT_temp/";
    #else
        std::string rootDirectory = std::filesystem::path(currentPath).parent_path().string() + "/ROOT_temp/";
    #endif

    if (!std::filesystem::exists(rootDirectory)) {std::filesystem::create_directory(rootDirectory);}

    primaryGenerator = static_cast < const PrimaryGenerator *> (G4RunManager::GetRunManager() -> GetUserPrimaryGeneratorAction()); 
    if (primaryGenerator && primaryGenerator -> GetParticleGun()) 
    {
        particle = primaryGenerator -> GetParticleGun() -> GetParticleDefinition();
        energy = primaryGenerator -> GetParticleGun() -> GetParticleEnergy();
        customRun -> SetPrimary(particle, energy);
    }   

    runID = thisRun -> GetRunID();
    directory = std::string(ROOT_OUTPUT_DIR);

    if (arguments == 1) {fileName = "/Sim_" + std::to_string(runID);}
    if (arguments == 2) {fileName = "/Sim_" + std::to_string(runID);}
    if (arguments == 3) {fileName = "/AttCoeff_" + std::to_string(runID);}
    if (arguments == 4) {fileName = "/Xray_" + std::to_string(runID);}
    if (arguments == 5) {fileName = "/CT_" + std::to_string(runID);}

    G4AnalysisManager * analysisManager = G4AnalysisManager::Instance();
    analysisManager -> SetFileName(directory + fileName);
    analysisManager -> OpenFile();

    const Run * currentRun = static_cast<const Run *>(thisRun);
    particleName = currentRun -> GetPrimaryParticleName();
    totalNumberOfEvents = currentRun -> GetNumberOfEventToBeProcessed();
    primaryEnergy = currentRun -> GetPrimaryEnergy();   

    simulationStartTime = std::chrono::system_clock::now();
    std::time_t now_start = std::chrono::system_clock::to_time_t(simulationStartTime);
    std::tm * now_tm_0 = std::localtime(&now_start);
    
    if (!isMaster && threadID == 0)
    {
        std::cout << std::endl;
        std::cout << "\033[32m================= RUN " << runID + 1 << " ==================" << std::endl;
        std::cout << "    The run is: " << totalNumberOfEvents << " " << particleName << " of " << G4BestUnit(primaryEnergy, "Energy") << std::endl;
        std::cout << "Start time: " << std::put_time(now_tm_0, "%H:%M:%S") << "    Date: " << std::put_time(now_tm_0, "%d-%m-%Y") << std::endl;
        std::cout << "\033[0m" << std::endl;
    }
}

void RunAction::EndOfRunAction(const G4Run * thisRun)
{  
    G4AnalysisManager * analysisManager = G4AnalysisManager::Instance();
    G4AccumulableManager * accumulableManager = G4AccumulableManager::Instance();
    accumulableManager -> Merge();
    MergeEnergySpectra();
    // G4cout << masterEnergySpectra.size() << G4endl;
    
    if (isMaster && arguments != 3) 
    { 
        detectorConstruction = static_cast <const DetectorConstruction*> (G4RunManager::GetRunManager() -> GetUserDetectorConstruction());   
        scoringVolumes = detectorConstruction -> GetAllScoringVolumes();

        totalMass = 0;
        index = 1;

        for (G4LogicalVolume * volume : scoringVolumes) 
        { 
            if (volume) {sampleMass = volume -> GetMass(); totalMass = totalMass + sampleMass;} 
            // G4cout << "Mass " << index << ": " << G4BestUnit(sampleMass, "Mass") << G4endl;
            index = index + 1;
        }
        
        const Run * currentRun = static_cast<const Run *>(thisRun);
        particleName = currentRun -> GetPrimaryParticleName();
        primaryEnergy = currentRun -> GetPrimaryEnergy();
        numberOfEvents = thisRun -> GetNumberOfEvent();

        TotalEnergyDeposit = fEdep.GetValue();
        radiationDose = TotalEnergyDeposit / totalMass;

        simulationEndTime = std::chrono::system_clock::now();
        now_end = std::chrono::system_clock::to_time_t(simulationEndTime);
        now_tm_1 = std::localtime(&now_end);
        
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(simulationEndTime - simulationStartTime);
        durationInSeconds = duration.count() * second;

        G4cout << G4endl; 
        G4cout << "\033[32mRun Summary:" << G4endl;
        G4cout << "--> Total mass of sample: " << G4BestUnit(totalMass, "Mass") << G4endl;
        G4cout << "--> Total energy deposition: " << G4BestUnit(TotalEnergyDeposit, "Energy") << G4endl;
        G4cout << "--> Radiation dose : " << G4BestUnit(radiationDose, "Dose") << G4endl;
        G4cout << G4endl;
        G4cout << "Ending time: " << std::put_time(now_tm_1, "%H:%M:%S") << "   Date: " << std::put_time(now_tm_1, "%d-%m-%Y") << G4endl;
        G4cout << "Total simulation time: " << G4BestUnit(durationInSeconds, "Time") << G4endl;
        G4cout << "========================================== \033[0m" << G4endl;
        G4cout << G4endl;
        
        if (arguments == 5)
        {
            analysisManager -> FillNtupleDColumn(1, 0, numberOfEvents);
            
            if (masterEnergySpectra.size() > 0)  {photonsEnergy = masterEnergySpectra;}
            if (masterEnergySpectra.size() <= 0) {photonsEnergy.push_back(primaryEnergy/keV);}
            
            analysisManager -> FillNtupleDColumn(1, 2, totalMass/kg);
            analysisManager -> FillNtupleDColumn(1, 3, TotalEnergyDeposit);
            analysisManager -> FillNtupleDColumn(1, 4, radiationDose);
            analysisManager -> AddNtupleRow(1);
        }
    }

    if (isMaster) {customRun -> EndOfRun();}

    analysisManager -> Write();
    analysisManager -> CloseFile();
    
    if (isMaster && arguments > 1) {MergeRootFiles(fileName);}
}

void RunAction::MergeEnergySpectra()
{
    primaryGenerator = static_cast <const PrimaryGenerator*> (G4RunManager::GetRunManager() -> GetUserPrimaryGeneratorAction());
    
    if (primaryGenerator) {energySpectra = primaryGenerator -> GetEnergySpectra();}

    G4MUTEXLOCK(&mergeMutex);  
    masterEnergySpectra.insert(masterEnergySpectra.end(), energySpectra.begin(), energySpectra.end());
    G4MUTEXUNLOCK(&mergeMutex); 
}

void RunAction::MergeRootFiles(const std::string & fileName) 
{
    std::string currentPath = std::filesystem::current_path().string();

    #ifdef __APPLE__
        std::string rootDirectory = std::filesystem::path(currentPath).string() + "/ROOT_temp/";
        std::string outputDirectory = std::filesystem::path(currentPath).string() + "/ROOT";
    #else
        std::string rootDirectory = std::filesystem::path(currentPath).parent_path().string() + "\\ROOT_temp\\";
        std::string outputDirectory = std::filesystem::path(currentPath).string() + "\\ROOT\\";
    #endif

    if (!std::filesystem::exists(outputDirectory)) {std::filesystem::create_directory(outputDirectory);}

    int fileIndex = 0;
    std::string mergedFileName;
    do 
    {
        mergedFileName = outputDirectory + fileName + std::to_string(fileIndex) + ".root";
        fileIndex++;
    } 
    while (std::filesystem::exists(mergedFileName));

    std::string haddCommand = "hadd -f -v 0 " + mergedFileName;
    
    for (const auto& entry : std::filesystem::directory_iterator(rootDirectory)) 
    {
        if (entry.is_regular_file() && entry.path().extension() == ".root") 
        {haddCommand += " " + entry.path().string();}
    }

    if (std::system(haddCommand.c_str()) == 0) 
    {
        G4cout << "~ Successfully merged ROOT files using hadd ~" << G4endl;
        std::filesystem::remove_all(rootDirectory);
    } 
    else {G4cerr << "Error: ROOT files merging with hadd failed!" << G4endl;}
}