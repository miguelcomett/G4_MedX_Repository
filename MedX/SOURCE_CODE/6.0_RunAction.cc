#include "6.0_RunAction.hh"
#include "8.0_SteppingAction.hh"

G4Mutex Mutex_Spectra = G4MUTEX_INITIALIZER;
std::map<G4float, G4int> masterEnergySpectra;

G4Mutex Mutex_EDep = G4MUTEX_INITIALIZER;
std::map<G4String, G4double> masterEnergyDeposition;

RunAction::RunAction()
{
    new G4UnitDefinition("milligray", "milliGy" , "Dose", milligray);
    new G4UnitDefinition("microgray", "microGy" , "Dose", microgray);
    new G4UnitDefinition("nanogray" , "nanoGy"  , "Dose", nanogray);
    new G4UnitDefinition("picogray" , "picoGy"  , "Dose", picogray);

    accumulableManager = G4AccumulableManager::Instance();
    accumulableManager -> RegisterAccumulable(EDepSum);

    analysisManager = G4AnalysisManager::Instance();
    analysisManager -> SetDefaultFileType("root");
    analysisManager -> SetVerboseLevel(0);

    if (arguments == 1 || arguments == 2)
    {
        analysisManager -> CreateNtuple("Hits", "Hits");
        analysisManager -> CreateNtupleDColumn("X");
        analysisManager -> CreateNtupleDColumn("Y");
        analysisManager -> CreateNtupleDColumn("X_det");
        analysisManager -> CreateNtupleDColumn("Y_det");
        analysisManager -> FinishNtuple(0);

        analysisManager -> CreateNtuple("Run Summary", "Run Summary");
        analysisManager -> CreateNtupleDColumn("Number_of_Photons");
        analysisManager -> CreateNtupleDColumn("Sample_Mass_kg");
        analysisManager -> CreateNtupleDColumn("EDep_Value_TeV");
        analysisManager -> CreateNtupleDColumn("Radiation_Dose_uSv");
        analysisManager -> FinishNtuple(1);
        
        analysisManager -> CreateNtuple("Tissue Energy Dep", "Tissue Energy Dep");
        analysisManager -> CreateNtupleSColumn("Tissue");
        analysisManager -> CreateNtupleDColumn("EnergyDeposition_TeV");
        analysisManager -> FinishNtuple(2);

        analysisManager -> CreateNtuple("Energy Spectra keV", "Energy Spectra keV");
        analysisManager -> CreateNtupleFColumn("Energies");
        analysisManager -> CreateNtupleIColumn("Counts");
        analysisManager -> FinishNtuple(3);

        analysisManager -> CreateNtuple("Detected Photons", "Detected Photons");
        analysisManager -> CreateNtupleDColumn("Energies_keV");
        analysisManager -> CreateNtupleDColumn("Wavelengths_nm");
        analysisManager -> FinishNtuple(4);
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
        
        analysisManager -> CreateNtuple("Radiation Dose", "Radiation Dose");
        analysisManager -> CreateNtupleSColumn("Tissue");
        analysisManager -> CreateNtupleDColumn("Radiation_Dose_uSv");
        analysisManager -> FinishNtuple(2);

        analysisManager -> CreateNtuple("Energy Spectra keV", "Energy Spectra keV");
        analysisManager -> CreateNtupleFColumn("Energies");
        analysisManager -> CreateNtupleIColumn("Counts");
        analysisManager -> FinishNtuple(3);
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

    currentRun = static_cast <const Run *> (thisRun);
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
        std::cout << "\033[32m" << "================= RUN " << runID + 1 << " ==================" << std::endl;
        std::cout << "    The run is: " << "\033[1m" << totalNumberOfEvents << " " << particleName << "\033[22m" << " of ";

        std::cout << "\033[1m";
        if (GunMode == 0) {std::cout << G4BestUnit(primaryEnergy, "Energy") << std::endl;}
        if (GunMode  > 0) {std::cout << primaryEnergy << " kVp" << std::endl;};
        std::cout << "\033[22m";

        std::cout << "Start time: " << std::put_time(now_tm_0, "%H:%M:%S") << "    Date: " << std::put_time(now_tm_0, "%d-%m-%Y") << "\033[0m" << std::endl;
        std::cout << std::endl;
    }
}

void RunAction::EndOfRunAction(const G4Run * thisRun)
{  
    accumulableManager -> Merge();
    MergeDataToMaster();

    if (isMaster) 
    { 
        if (arguments == 1 || arguments == 2 || arguments == 5)
        {   
            scoringVolumes = detectorConstruction -> GetAllScoringVolumes();

            for (G4LogicalVolume * volume : scoringVolumes) 
            {
                if (volume)
                {
                    sampleMass = volume -> GetMass(); 
                    totalMass = totalMass + sampleMass;
                } 
            }
            
            particleName = currentRun -> GetPrimaryParticleName();
            primaryEnergy = currentRun -> GetPrimaryEnergy();
            numberOfEvents = thisRun -> GetNumberOfEvent();

            TotalEnergyDeposit = EDepSum.GetValue();
            radiationDose = TotalEnergyDeposit / totalMass;

            G4cout << G4endl; 
            G4cout << "\033[32m" << "Run Summary:" << G4endl;
            G4cout << "--> Total Mass of Sample: " << "\033[1m" << G4BestUnit(totalMass, "Mass")            << "\033[22m" << G4endl;
            G4cout << "--> Energy Deposition: "    << "\033[1m" << G4BestUnit(TotalEnergyDeposit, "Energy") << "\033[22m" << G4endl;
            G4cout << G4endl;
            G4cout << "--> Radiation Dose: "       << "\033[1m" << G4BestUnit(radiationDose, "Dose")        << "\033[22m" << G4endl;

            totalMass = totalMass / kg;
            TotalEnergyDeposit = TotalEnergyDeposit / TeV;
            radiationDose = radiationDose / microgray;
            primaryEnergy = primaryEnergy / keV;

            analysisManager -> FillNtupleDColumn(1, 0, numberOfEvents);
            analysisManager -> FillNtupleDColumn(1, 1, totalMass);
            analysisManager -> FillNtupleDColumn(1, 2, TotalEnergyDeposit);
            analysisManager -> FillNtupleDColumn(1, 3, radiationDose);
            analysisManager -> AddNtupleRow(1);
            
            if (masterEnergyDeposition.size() > 0)
            {   
                for (const auto & entry : masterEnergyDeposition) 
                {
                    tissueName = entry.first;
                    tissueEDep = entry.second;

                    for (auto & volume : scoringVolumes) 
                    {
                        if (volume -> GetName() == tissueName) {sampleMass = volume -> GetMass(); break;}
                    }

                    radiationDose = tissueEDep / sampleMass;

                    G4cout 
                    << "  > [" << std::setw(7) << std::left << tissueName << "] "
                    << "Mass: " 
                    << "\033[1m" << std::setw(8) << std::left << std::setprecision(5) << sampleMass/kg << " kg\033[22m"
                    << ", Energy Deposition: " 
                    << "\033[1m" << std::setw(6) << std::left << std::setprecision(5) << G4BestUnit(tissueEDep, "Energy") << "\033[22m"
                    << " === Radiation Dose: " 
                    << "\033[1m" << std::setw(9) << std::left << std::setprecision(6) << G4BestUnit(radiationDose, "Dose") << "\033[22m" <<
                    G4endl;

                    analysisManager -> FillNtupleSColumn(2, 0, tissueName); // tissueName.c_str()
                    analysisManager -> FillNtupleDColumn(2, 1, radiationDose / microgray);
                    analysisManager -> AddNtupleRow(2);
                }
            }

            if (masterEnergySpectra.size() == 0) // mono
            {
                analysisManager -> FillNtupleFColumn(3, 0, primaryEnergy);
                analysisManager -> FillNtupleIColumn(3, 1, 1);
                analysisManager -> AddNtupleRow(3);
            }
            if (masterEnergySpectra.size() > 0) // poly
            {                
                for (const auto & entry : masterEnergySpectra) 
                {
                    energies = entry.first;
                    frequency = entry.second;

                    analysisManager -> FillNtupleFColumn(3, 0, energies);
                    analysisManager -> FillNtupleIColumn(3, 1, frequency);
                    analysisManager -> AddNtupleRow(3);
                }
            }

            simulationEndTime = std::chrono::system_clock::now();
            now_end = std::chrono::system_clock::to_time_t(simulationEndTime);
            now_tm_1 = std::localtime(&now_end);
            
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(simulationEndTime - simulationStartTime);
            durationInSeconds = duration.count() * second;

            G4cout << G4endl;
            G4cout << "Ending time: " << std::put_time(now_tm_1, "%H:%M:%S") << "   Date: " << std::put_time(now_tm_1, "%d-%m-%Y") << G4endl;
            G4cout << "Total simulation time: " << G4BestUnit(durationInSeconds, "Time") << G4endl;
            G4cout << "========================================== \033[0m" << G4endl;
            G4cout << G4endl;
        }
        
        customRun -> EndOfRun();
    }

    analysisManager -> Write();
    analysisManager -> CloseFile();
    
    if (isMaster && arguments > 1) {MergeRootFiles(baseName, tempDirectory, rootDirectory);}
}

void RunAction::MergeDataToMaster()
{
    if (primaryGenerator) {energyHistogram = primaryGenerator -> GetEnergySpectra();}
    
    G4MUTEXLOCK(&Mutex_Spectra);
    for (const auto & entry : energyHistogram) {masterEnergySpectra[entry.first] += entry.second;}
    G4MUTEXUNLOCK(&Mutex_Spectra);

    const G4UserSteppingAction * userSteppingAction = G4RunManager::GetRunManager() -> GetUserSteppingAction();
    const SteppingAction * steppingAction = dynamic_cast<const SteppingAction*> (userSteppingAction); 
    
    if (steppingAction){energyDepositionMap = steppingAction -> GetEnergyDepositionMap();}
    
    G4MUTEXLOCK(&Mutex_EDep);
    for (const auto & entry : energyDepositionMap) {masterEnergyDeposition[entry.first] += entry.second;}
    G4MUTEXUNLOCK(&Mutex_EDep);
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