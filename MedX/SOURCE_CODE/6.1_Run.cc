#include "6.0_RunAction.hh"
#include "6.1_Run.hh"

Run::Run(){}
Run::~Run(){}

void Run::SetPrimary(G4ParticleDefinition * particle, G4double energy)
{ 
    link_ParticleDefinition = particle; 
    link_Energy = energy;
}

G4double Run::GetPrimaryEnergy() const {return link_Energy;}
G4String Run::GetPrimaryParticleName() const {return link_ParticleDefinition ? link_ParticleDefinition -> GetParticleName() : "Unknown";}

void Run::CountProcesses(G4String processName) 
{
    if (arguments == 3)
    {
        std::map <G4String, G4int> ::iterator it = processCounter.find(processName);
        if ( it == processCounter.end()) {processCounter[processName] = 1;} else {processCounter[processName]++;}
    }
}

void Run::EndOfRun()
{
    if (arguments == 3)
    {
        detectorConstruction = static_cast < const DetectorConstruction *> (G4RunManager::GetRunManager() -> GetUserDetectorConstruction());     
        
        material = detectorConstruction -> GetMaterial();
        density = material  -> GetDensity();
        thickness = detectorConstruction -> GetThickness();
        particleName = link_ParticleDefinition -> GetParticleName(); 

        G4cout << G4endl; G4cout << G4endl;
        G4cout << "============== Run Summary ===============" << G4endl;
        G4cout << "     The run is: " << numberOfEvent << " " << particleName << " of "<< G4BestUnit(link_Energy, "Energy") << G4endl;
        G4cout << "      Through " << G4BestUnit(thickness, "Length") << "of " << material -> GetName() << G4endl;
        G4cout << "        (Density: " << G4BestUnit(density, "Volumic Mass") << ")" << G4endl;
        G4cout << "==========================================" << G4endl;
        G4cout << G4endl; G4cout << G4endl;

        G4cout << "======== Process calls frequency ========" << G4endl;

        totalCount = 0;
        survive = 0; 

        std::map < G4String, G4int >::iterator it;  
        for (it = processCounter.begin(); it != processCounter.end(); it++) 
        {
            processName = it -> first;
            count = it -> second;
            totalCount = totalCount + count; 
            G4cout << processName << " = " << count << G4endl;
            if (processName == "Transportation") survive = count;
        }
        processCounter.clear();

        G4cout << "==========================================" << G4endl;
        G4cout << G4endl;

        digits = 4; defaultDecimals = G4cout.precision(digits);
        if (totalCount == 0) { G4cout.precision(defaultDecimals); return;};  

        ratio = double(survive) / totalCount;
        crossSection = - std::log(ratio) / thickness;     
        Coefficient = crossSection / density;
        link_Energy = link_Energy / keV;
    
        G4cout << G4endl;
        G4cout << "======== Transportation Summary ==========" << G4endl;
        G4cout << "Number of particles unaltered after " << G4BestUnit(thickness,"Length") << G4endl;
        G4cout << "--> Of " << material -> GetName() << ", p = " << G4BestUnit(density, "Volumic Mass") << G4endl;
        G4cout << "--> " << survive << " over " << totalCount << " incident particles." << G4endl;
        G4cout << "--> Ratio = " << 100 * ratio << "%" << G4endl;
        G4cout << "--> CrossSection per volume: " << crossSection*cm << " cm^-1" << G4endl;
        G4cout << "--> CrossSection per mass: " << G4BestUnit(Coefficient, "Surface/Mass") << G4endl;
        G4cout << "==========================================" << G4endl;
        G4cout << G4endl; G4cout << G4endl;
        
        Coefficient = Coefficient * g / cm2;

        G4AnalysisManager * analysisManager = G4AnalysisManager::Instance();
        analysisManager -> FillNtupleDColumn(0, 0, Coefficient);
        analysisManager -> FillNtupleDColumn(0, 1, link_Energy);
        analysisManager -> FillNtupleDColumn(0, 2, survive);
        analysisManager -> AddNtupleRow(0);
        
        G4cout.precision(defaultDecimals);
    }   
}

void Run::Merge(const G4Run * run)
{
    const Run * localRun = static_cast <const Run*> (run);

    link_ParticleDefinition = localRun -> link_ParticleDefinition;
    link_Energy = localRun -> link_Energy;
            
    std::map<G4String, G4int>::const_iterator it;
    for (it  = localRun -> processCounter.begin(); it != localRun -> processCounter.end(); ++it) 
    {
        processName = it -> first;
        localCount  = it -> second;

        if ( processCounter.find(processName) == processCounter.end()) {processCounter[processName] = localCount;} 
        else {processCounter[processName] += localCount;}         
    }
        
    G4Run::Merge(run);  
} 