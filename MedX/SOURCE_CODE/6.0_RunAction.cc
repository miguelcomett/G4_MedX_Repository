#include "6.0_RunAction.hh"

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
    }

    if (arguments == 5)
    {
        analysisManager -> CreateNtuple("Hits", "Hits");
        analysisManager -> CreateNtupleDColumn("x_ax");
        analysisManager -> CreateNtupleDColumn("y_ax");
        analysisManager -> FinishNtuple(0);

        analysisManager -> CreateNtuple("Run Summary", "Run Summary");
        analysisManager -> CreateNtupleDColumn("Number_of_Photons");
        analysisManager -> CreateNtupleDColumn("Initial_Energy_keV");
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
    
    if (isMaster && arguments != 3) 
    { 
        detectorConstruction = static_cast < const DetectorConstruction *> (G4RunManager::GetRunManager() -> GetUserDetectorConstruction());   
        std::vector <G4LogicalVolume*> scoringVolumes = detectorConstruction -> GetAllScoringVolumes();

        totalMass = 0;
        index = 1;

        for (G4LogicalVolume * volume : scoringVolumes) 
        { 
            if (volume) {sampleMass = volume -> GetMass(); totalMass = totalMass + sampleMass;} // G4cout << "Mass " << index << ": " << G4BestUnit(sampleMass, "Mass") << G4endl;
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
    }
    
    if (arguments == 5) 
    {   
        primaryEnergy = primaryEnergy / keV;
        totalMass = totalMass / kg;
        TotalEnergyDeposit = TotalEnergyDeposit / TeV;
        radiationDose = radiationDose / microgray;

        analysisManager -> FillNtupleDColumn(1, 0, numberOfEvents);
        analysisManager -> FillNtupleDColumn(1, 1, primaryEnergy);
        analysisManager -> FillNtupleDColumn(1, 2, totalMass);
        analysisManager -> FillNtupleDColumn(1, 3, TotalEnergyDeposit);
        analysisManager -> FillNtupleDColumn(1, 4, radiationDose);
        analysisManager -> AddNtupleRow(1);
    }

    if (isMaster) {customRun -> EndOfRun();}

    analysisManager -> Write();
    analysisManager -> CloseFile();
    
    if (isMaster && arguments > 1) {MergeRootFiles();}
}

void RunAction::MergeRootFiles()
{
    TFileMerger merger;
    merger.SetFastMethod(true);

    std::string currentPath = std::filesystem::current_path().string();

    #ifdef __APPLE__
        std::string rootDirectory = std::filesystem::path(currentPath).string() + "/ROOT_temp/";
        std::string outputDirectory = std::filesystem::path(currentPath).string() + "/ROOT/";
    #else
        std::string rootDirectory = std::filesystem::path(currentPath).parent_path().string() + "\\ROOT_temp\\";
        std::string outputDirectory = std::filesystem::path(currentPath).string() + "\\ROOT\\";
    #endif
   
    if (!std::filesystem::exists(outputDirectory)) {std::filesystem::create_directory(outputDirectory);}

    std::string fileName;  // Definir el nombre base del archivo según el valor de 'arguments'
    if (arguments == 1 || arguments == 2) {fileName = "Sim_";}
    else if (arguments == 3) {fileName = "AttCoeff_";}
    else if (arguments == 4) {fileName = "Rad_";}
    else if (arguments == 5) {fileName = "CT_";}

    // Encontrar el primer índice disponible para el archivo de salida
    int fileIndex = 0; std::string mergedFileName;
    do {mergedFileName = outputDirectory + fileName + std::to_string(fileIndex) + std::to_string(runID) + ".root"; fileIndex++;} 
    while (std::filesystem::exists(mergedFileName));

    // Iterar sobre los archivos en el directorio ROOT y agregar archivos .root al merger
    for (const auto & entry : std::filesystem::directory_iterator(rootDirectory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".root")
        {
            std::string filePath = entry.path().string();
            merger.AddFile(filePath.c_str(), false);
            std::filesystem::remove(entry.path()); // Eliminar el archivo después de agregarlo al merger
        }
    }

    merger.OutputFile(mergedFileName.c_str()); // Guardar el archivo fusionado en la nueva carpeta Output

    if (merger.Merge()) {RemoveJunkDataFromRoot(mergedFileName); G4cout << "~ Successfully merged ROOT files ~" << G4endl; G4cout << G4endl;}
    else {G4cout << "Error during ROOT file merging!" << G4endl;}

    std::filesystem::remove_all(rootDirectory);
}

void RunAction::RemoveJunkDataFromRoot(const std::string & mergedFileName)
{
    TFile * mergedFile = TFile::Open(mergedFileName.c_str(), "UPDATE");
    if (!mergedFile || mergedFile -> IsZombie()) {return;}

    TTree * tree = dynamic_cast<TTree*>(mergedFile -> Get("Run Summary")); // Obtener el arbol del archivo
    if (!tree) {mergedFile -> Close(); return;}

    double numberOfPhotons, initialEnergy, sampleMass, edepValue, radiationDose;  // Variables para almacenar los datos de las columnas

    tree -> SetBranchAddress("Number_of_Photons",  & numberOfPhotons);
    tree -> SetBranchAddress("Initial_Energy_keV", & initialEnergy);
    tree -> SetBranchAddress("Sample_Mass_kg",     & sampleMass);
    tree -> SetBranchAddress("EDep_Value_TeV",     & edepValue);
    tree -> SetBranchAddress("Radiation_Dose_uSv", & radiationDose);

    // Inicializa las variables para los valores maximos
    double maxNumberOfPhotons = -DBL_MAX;
    double maxInitialEnergy   = -DBL_MAX;
    double maxSampleMass      = -DBL_MAX;
    double maxEdepValue       = -DBL_MAX;
    double maxRadiationDose   = -DBL_MAX;

    Long64_t maxEntryIndex = -1; // Para almacenar el indice de la entrada con el valor maximo
    TTree * newTree = tree -> CloneTree(0); // Creamos un nuevo arbol vacio para almacenar las entradas validas

    for (Long64_t i = 0; i < tree -> GetEntries(); ++i) 
    {
        tree -> GetEntry(i);
        if (numberOfPhotons == 0 || initialEnergy == 0 || sampleMass == 0 || edepValue == 0 || radiationDose == 0) {continue;} // Comprobar si alguno de los valores es cero y, si es asi, no agregarlo
        
        if (numberOfPhotons > maxNumberOfPhotons) // Comparar y actualizar los valores maximos
        {
            maxNumberOfPhotons = numberOfPhotons;
            maxInitialEnergy = initialEnergy;
            maxSampleMass = sampleMass;
            maxEdepValue = edepValue;
            maxRadiationDose = radiationDose;
            maxEntryIndex = i; // Guardamos el indice de la entrada con los valores maximos
        }
        newTree -> Fill();
    }

    if (maxEntryIndex == -1) {mergedFile -> Close(); return;} // G4cout << "Error: No valid entries found in the tree." << G4endl;
    TTree * maxTree = tree -> CloneTree(0); // Crear un nuevo arbol vacio con la misma estructura
    tree -> GetEntry(maxEntryIndex); // Obtener la entrada con el valor maximo

    maxTree -> SetBranchAddress("Number_of_Photons",  & maxNumberOfPhotons); // Establecer las ramas del nuevo arbol con los valores maximos
    maxTree -> SetBranchAddress("Initial_Energy_keV", & maxInitialEnergy);
    maxTree -> SetBranchAddress("Sample_Mass_kg",     & maxSampleMass);
    maxTree -> SetBranchAddress("EDep_Value_TeV",     & maxEdepValue);
    maxTree -> SetBranchAddress("Radiation_Dose_uSv", & maxRadiationDose);

    maxTree -> Fill();    
    maxTree -> Write("Run Summary", TObject::kOverwrite); // Sobrescribir el arbol original con el nuevo arbol que solo tiene la entrada maxima
    mergedFile -> Close();
}