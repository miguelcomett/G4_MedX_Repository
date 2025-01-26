#include "6.0_RunAction.hh"

G4Mutex mergeMutex = G4MUTEX_INITIALIZER;
std::map<G4float, G4int> masterEnergySpectra;

RunAction::RunAction()
{
    new G4UnitDefinition("milligray", "milliGy" , "Dose", milligray);
    new G4UnitDefinition("microgray", "microGy" , "Dose", microgray);
    new G4UnitDefinition("nanogray" , "nanoGy"  , "Dose", nanogray);
    new G4UnitDefinition("picogray" , "picoGy"  , "Dose", picogray);

    accumulableManager = G4AccumulableManager::Instance();
    accumulableManager -> RegisterAccumulable(fEdep);

    analysisManager = G4AnalysisManager::Instance();
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
        analysisManager -> CreateNtupleDColumn("Sample_Mass_kg");
        analysisManager -> CreateNtupleDColumn("EDep_Value_TeV");
        analysisManager -> CreateNtupleDColumn("Radiation_Dose_uSv");
        analysisManager -> FinishNtuple(1);
        
        analysisManager -> CreateNtuple("Energy Spectra keV", "Energy Spectra keV");
        analysisManager -> CreateNtupleFColumn("Energies");
        analysisManager -> CreateNtupleIColumn("Counts");
        analysisManager -> FinishNtuple(2);
    }
}

RunAction::~RunAction(){}

G4Run * RunAction::GenerateRun() {customRun = new Run(); return customRun;}

void RunAction::BeginOfRunAction(const G4Run * thisRun)
{
    accumulableManager -> Reset();
    masterEnergySpectra.clear();

    currentPath = std::filesystem::current_path().string();

    #ifdef __APPLE__
        tempDirectory = std::filesystem::path(currentPath).string() + "/ROOT_temp/";
        rootDirectory = std::filesystem::path(currentPath).string() + "/ROOT";
    #else
        tempDirectory = std::filesystem::path(currentPath).parent_path().string() + "/ROOT_temp";
        rootDirectory = std::filesystem::path(currentPath).string() + "\\ROOT\\";
    #endif

    if (!std::filesystem::exists(tempDirectory)) {std::filesystem::create_directory(tempDirectory);}
    if (!std::filesystem::exists(rootDirectory)) {std::filesystem::create_directory(rootDirectory);}

    if (arguments == 1) {baseName = "/Sim";}
    if (arguments == 2) {baseName = "/Sim";}
    if (arguments == 3) {baseName = "/AttCoeff";}
    if (arguments == 4) {baseName = "/Xray";}
    if (arguments == 5) {baseName = "/CT";}

    runID = thisRun -> GetRunID();
    fileName = baseName + std::to_string(runID);

    analysisManager -> SetFileName(tempDirectory + baseName);
    analysisManager -> OpenFile();
    
    if (primaryGenerator) 
    {
        particle = primaryGenerator -> GetParticleGun() -> GetParticleDefinition();
        energy = primaryGenerator -> GetParticleGun() -> GetParticleEnergy();
        customRun -> SetPrimary(particle, energy);
    }

    currentRun = static_cast<const Run *>(thisRun);
    particleName = currentRun -> GetPrimaryParticleName();
    totalNumberOfEvents = currentRun -> GetNumberOfEventToBeProcessed();
    primaryEnergy = currentRun -> GetPrimaryEnergy();   
    
    if (primaryGenerator) {GunMode = primaryGenerator -> GetGunMode();} 
    if (GunMode == 1) {primaryEnergy = 80;}
    if (GunMode == 2) {primaryEnergy = 140;}

    simulationStartTime = std::chrono::system_clock::now();
    now_start = std::chrono::system_clock::to_time_t(simulationStartTime);
    now_tm_0 = std::localtime(& now_start);
    
    threadID = G4Threading::G4GetThreadId();

    if (threadID == 0)
    {
        std::cout << std::endl;
        std::cout << "\033[32m================= RUN " << runID + 1 << " ==================" << std::endl;
        std::cout << "    The run is: " << totalNumberOfEvents << " " << particleName << " of " ;
        
        if (GunMode == 0) {std::cout << G4BestUnit(primaryEnergy, "Energy") << std::endl;}
        if (GunMode  > 0) {std::cout << primaryEnergy << " kVp" << std::endl;}

        std::cout << "Start time: " << std::put_time(now_tm_0, "%H:%M:%S") << "    Date: " << std::put_time(now_tm_0, "%d-%m-%Y") << std::endl;
        std::cout << "\033[0m" << std::endl;
    }
}

void RunAction::EndOfRunAction(const G4Run * thisRun)
{  
    accumulableManager -> Merge();
    MergeEnergySpectra();

    if (isMaster) 
    { 
        if (arguments != 3)
        {
            scoringVolumes = detectorConstruction -> GetAllScoringVolumes();

            totalMass = 0;
            index = 1;

            for (G4LogicalVolume * volume : scoringVolumes) 
            { 
                if (volume) {sampleMass = volume -> GetMass(); totalMass = totalMass + sampleMass;} 
                // G4cout << "Mass " << index << ": " << G4BestUnit(sampleMass, "Mass") << G4endl;
                index = index + 1;
            }
            
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
        }
        
        if (arguments == 5)
        {   
            totalMass = totalMass / kg;
            TotalEnergyDeposit = TotalEnergyDeposit / TeV;
            radiationDose = radiationDose / microgray;
            primaryEnergy = primaryEnergy / keV;

            analysisManager -> FillNtupleDColumn(1, 0, numberOfEvents);
            analysisManager -> FillNtupleDColumn(1, 1, totalMass);
            analysisManager -> FillNtupleDColumn(1, 2, TotalEnergyDeposit);
            analysisManager -> FillNtupleDColumn(1, 3, radiationDose);
            analysisManager -> AddNtupleRow(1);
            
            if (masterEnergySpectra.size() == 0) 
            {
                frequency = 1;
                analysisManager -> FillNtupleFColumn(2, 0, primaryEnergy);
                analysisManager -> FillNtupleIColumn(2, 1, frequency);
                analysisManager -> AddNtupleRow(2);
            }
            
            if (masterEnergySpectra.size() > 0)
            {
                energies = 0.0;
                frequency = 0;
                
                for (const auto & entry : masterEnergySpectra) 
                {
                    energies = entry.first;
                    frequency = entry.second;

                    analysisManager -> FillNtupleFColumn(2, 0, energies);
                    analysisManager -> FillNtupleIColumn(2, 1, frequency);
                    analysisManager -> AddNtupleRow(2);
                }
            }
        }
        
        customRun -> EndOfRun();
    }

    analysisManager -> Write();
    analysisManager -> CloseFile();
    
    if (isMaster && arguments > 1) {MergeRootFiles(baseName, tempDirectory, rootDirectory);}
}

void RunAction::MergeEnergySpectra()
{
    if (primaryGenerator) {energyHistogram = primaryGenerator -> GetEnergySpectra();}
    
    G4MUTEXLOCK(&mergeMutex);
    for (const auto & entry : energyHistogram) {masterEnergySpectra[entry.first] += entry.second;}
    G4MUTEXUNLOCK(&mergeMutex);
}

void RunAction::MergeRootFiles(const std::string & baseName, const std::string & tempDirectory, const std::string & rootDirectory) 
{   
    fileIndex = 0;
    mergedFileName = rootDirectory + baseName + "_" + std::to_string(fileIndex) + std::to_string(runID) + ".root";
    while (std::filesystem::exists(mergedFileName))
    {
        fileIndex += 1;
        mergedFileName = rootDirectory + baseName + "_" + std::to_string(fileIndex) + std::to_string(runID) + ".root";
    } 

    haddCommand = "hadd -f -v 0 " + mergedFileName;
    
    for (const auto & entry : std::filesystem::directory_iterator(tempDirectory)) 
    {
        if (entry.is_regular_file() && entry.path().extension() == ".root") {haddCommand += " " + entry.path().string();}
    }

    if (std::system(haddCommand.c_str()) == 0) 
    {
        G4cout << "~ Successfully Merged Root Files." << G4endl; 
        G4cout << "~ File written: " << mergedFileName << G4endl;
        G4cout << G4endl;
        std::filesystem::remove_all(tempDirectory);
    } 
    else {G4cerr << "Error: ROOT files merging with hadd failed!" << G4endl;}
}