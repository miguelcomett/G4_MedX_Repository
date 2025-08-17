#include "5.0_PrimaryGenerator.hh"

PrimaryGenerator::PrimaryGenerator()
{
    particleGun = new G4ParticleGun();
    particleTable = G4ParticleTable::GetParticleTable();
    particleName = particleTable -> FindParticle("gamma");
    particleGun -> SetParticleDefinition(particleName);   
    particleGun -> SetParticleEnergy(40 * keV); 

    GeneratorMessenger = new PrimaryGeneratorMessenger(this);

    SpectraMode = 0;
    Xpos =  000.0 * mm;
    Ypos = -025.0 * mm;
    Zpos = -450.0 * mm;
    
    SpanX = 250.0 * mm;
    SpanY = 270.0 * mm;

    GunAngle = 0.0;

    Xcos = false;
    
    threadID = G4Threading::G4GetThreadId();
    if (threadID == 0) {std::cout << std::endl; std::cout << "------------- GUN MESSENGERS -------------" << std::endl;}
}

PrimaryGenerator::~PrimaryGenerator() {delete particleGun; delete GeneratorMessenger;}

void PrimaryGenerator::GeneratePrimaries(G4Event * anEvent)
{ 
    if (SpectraMode == 1 || SpectraMode == 2) 
    {
        RealEnergy = InverseCumul(); 
        particleGun -> SetParticleEnergy(RealEnergy);
        
        totalEnergy = totalEnergy + RealEnergy;

        RealEnergy = RealEnergy / keV;
        
        Decimals = 1;
        roundingScale = std::pow(10, Decimals);
        RealEnergy = std::round(RealEnergy * roundingScale) / roundingScale;
        
        energyHistogram[RealEnergy]++;
    }

    if (Xgauss == true) 
    {
        x0 = G4RandGauss::shoot(0, SpanX / 2.5); // Origin, Std Deviation (Sigma) 
        while (x0 > SpanX || x0 < -SpanX) {x0 = G4RandGauss::shoot(0, SpanX);}
    }
    if (Xgauss == false) 
    {
        x0 = 2 * (G4UniformRand() - 0.5); 
        x0 = x0 * SpanX;
    }
    
    if (Xcos == true) 
    {
        if (detectorConstruction) {thoraxAngle = detectorConstruction -> GetThoraxAngle();} else {thoraxAngle = 0;}
        
        if (thoraxAngle < 90)  {gunAngle = thoraxAngle;}
        if (thoraxAngle >= 90) {gunAngle = thoraxAngle - 180;}
        if (thoraxAngle > 270) {gunAngle = thoraxAngle - 360;}
        gunAngle = gunAngle * (2*pi / 360);
        x0 = x0 * std::cos(gunAngle/2);

        model_width = 230; 
        model_depth = 150;
        minimum_span = model_depth / model_width;
        minimum_span = minimum_span * 1.15; // padding for safety
        x0 = x0 * ( (std::cos(gunAngle) * std::cos(gunAngle)) * (1-minimum_span) + minimum_span);
    }

    x0 = x0 + Xpos; 
    
    y0 = 2 * (G4UniformRand() - 0.5);
    y0 = y0 * SpanY;
    y0 = y0 + Ypos;

    z0 = Zpos; 

    photonPosition = G4ThreeVector(x0, y0, z0);
    particleGun -> SetParticlePosition(photonPosition);

    AngleInCarts = std::tan(GunAngle * (2*pi / 360.0));
    Theta = AngleInCarts * (G4UniformRand() - 0.5) * 2;
    Phi   = AngleInCarts * (G4UniformRand() - 0.5) * 2;
    
    photonMomentum = G4ThreeVector(Theta, Phi, -1.0);
    particleGun -> SetParticleMomentumDirection(photonMomentum);

    particleGun -> GeneratePrimaryVertex(anEvent);
}

// Create Ratiation Spectra ====================================================================================================================

void PrimaryGenerator::ReadSpectrumFromFile(const std::string & filename, std::vector<G4double> & EnergyVector, std::vector<G4double> & IntensityVector, G4int & energyDataPoints) 
{ 
    std::ifstream spectraFile(filename);
    if (!spectraFile) {G4cerr << "Error opening file: " << filename << G4endl; return;}

    EnergyVector.clear();
    IntensityVector.clear();
    energyDataPoints = energy = intensity = 0.0;

    // G4cout << EnergyVector.size() << IntensityVector.size() << energyDataPoints << G4endl;

    while (spectraFile >> energy >> intensity) // Convertir energía de keV a las unidades internas de Geant4
    {
        EnergyVector.push_back(energy * keV);
        IntensityVector.push_back(intensity);
        energyDataPoints++; 
    }

    spectraFile.close();
}

void PrimaryGenerator::SpectraFunction() 
{
    // tabulated function, Y is assumed positive, linear per segment, continuous

    X_vector.clear();
    Y_vector.clear();
    Slopes_vector.clear();
    Y_Cumulative.clear();
    Y_max = 0.0;

    ReadSpectrumFromFile(spectrumFile, EnergyVector, IntensityVector, energyDataPoints);

    if (threadID == 0) 
    {
        // std::cout << "Energy Data Points Read: " << energyDataPoints << std::endl; std::cout << std::endl;
        
        for (size_t i = 0; i < EnergyVector.size(); ++i) 
        {
            // std::cout << 
            // "Energy: "    << std::fixed << std::setprecision(1) << EnergyVector[i] / keV << " keV   " <<  
            // "Intensity: " << std::fixed << std::setprecision(3) << IntensityVector[i] * 100 << std::endl;

            // for (int j=0; j<std::ceil(IntensityVector[i] * 1000); j++){energySpectra.push_back(EnergyVector[i] / keV);}
        }
    }

    X_vector.resize(energyDataPoints); 
    Y_vector.resize(energyDataPoints);
    
    for (G4int j=0; j<energyDataPoints; j++) 
    {
        X_vector[j] = EnergyVector[j]; 
        Y_vector[j] = IntensityVector[j]; 
        if (Y_max < Y_vector[j]) {Y_max = Y_vector[j];}
    }

    Slopes_vector.resize(energyDataPoints); 
    for (G4int j=0; j<energyDataPoints-1; j++) 
    {
        Slopes_vector[j] = (Y_vector[j+1] - Y_vector[j])/(X_vector[j+1] - X_vector[j]);
    }

    Y_Cumulative.resize(energyDataPoints);
    Y_Cumulative[0] = 0.0;
    for (G4int j=1; j<energyDataPoints; j++) 
    {
        Y_Cumulative[j] = Y_Cumulative[j-1] + 0.5 * (Y_vector[j] + Y_vector[j-1]) * (X_vector[j] - X_vector[j-1]);
    }     
}

G4double PrimaryGenerator::InverseCumul() 
{ 
    // Function to estimate counts --> cumulative function is second order polynomial

    X_random = Y_random = Alfa = Beta = Gamma = Delta = 0.0;

    Y_random = G4UniformRand() * Y_Cumulative[energyDataPoints-1]; 
 
    bins = energyDataPoints - 2; 
    while ( (Y_Cumulative[bins] > Y_random) && (bins > 0) ) {bins--;} // y_rndm --> x_rndm :  Y_Cumulative(x) is second order polynomial
    
    X_random = X_vector[bins];
    Alfa = Slopes_vector[bins];

    sign = 1;
    
    if (Alfa != 0.0) 
    {
        Beta = Y_vector[bins] / Alfa, 
        Gamma = 2 * (Y_random - Y_Cumulative[bins]) / Alfa;
        Delta = Beta * Beta + Gamma;
        
        if (Alfa < 0.0) {sign = -1;}
        
        X_random += sign * std::sqrt(Delta) - Beta;    
    } 
    else if (Y_vector[bins] > 0.0) 
    {
        X_random += (Y_random - Y_Cumulative[bins]) / Y_vector[bins];
    }
    
    return X_random;
}

// Messengers ==============================================================================================================================

void PrimaryGenerator::SetGunXpos(G4double newXpos)
{
    if (newXpos != Xpos) {Xpos = newXpos; 
        if (threadID == 0) {std::cout << "-> Source X Position changed to: " << Xpos << std::endl;} 
    else if (threadID == 0) {std::cout << "-> Same Position Selected." << std::endl;}}
}

void PrimaryGenerator::SetGunYpos(G4double newYpos)
{
    if (newYpos != Ypos) {Ypos = newYpos;
        if (threadID == 0) {std::cout << "-> Source Y Position changed to: " << Ypos << std::endl;} 
    else if (threadID == 0) {G4cout << "-> Same Position Selected." << std::endl;}}
}

void PrimaryGenerator::SetGunZpos(G4double newZpos)
{
    if (newZpos != Zpos) {Zpos = newZpos; 
        if (threadID == 0) {std::cout << "-> Source Z Position changed to: " << Zpos << std::endl;} 
    else if (threadID == 0) {std::cout << "-> Same Position Selected." << std::endl;}}
}

void PrimaryGenerator::SetGunXGauss(G4bool newXgauss)
{   
    if (newXgauss == true) {Xgauss = true; 
        if (threadID == 0) {std::cout << "-> Source X changed to: Gauss Distribution" << std::endl;}}
    if (newXgauss == false) {Xgauss = false; 
        if (threadID == 0) {std::cout << "-> Source X changed to: Linear Distribution" << std::endl;}}
}

void PrimaryGenerator::SetGunXcos(G4bool newXcos)
{   
    if (newXcos == true) {Xcos = true; 
        if (threadID == 0) {std::cout << "-> Source X span changed to: Cosine function " << std::endl;}}
    if (newXcos == false) {Xcos = false; 
        if (threadID == 0) {std::cout << "-> Source X span fixed" << std::endl;}}
}

void PrimaryGenerator::SetGunSpanX(G4double newSpanX)
{
    if (newSpanX != SpanX) {SpanX = newSpanX; 
        if (threadID == 0) {std::cout << "-> Source X Span changed to: " << SpanX << std::endl;} 
    else if (threadID == 0) {std::cout << "-> Same Span selected." << std::endl;}}
}

void PrimaryGenerator::SetGunSpanY(G4double newSpanY)
{
    if (newSpanY != SpanY) {SpanY = newSpanY; 
        if (threadID == 0) {std::cout << "-> Source Y Span changed to: " << SpanY << std::endl;}}
    else if (threadID == 0) {std::cout << "-> Same Span selected." << std::endl;}
}

void PrimaryGenerator::SetGunAngle(G4double newAngle)
{   
    if (newAngle != GunAngle) {GunAngle = newAngle; 
        if (threadID == 0) {std::cout << "-> Source Angle changed to: " << GunAngle << std::endl;} 
    else if (threadID == 0) {std::cout << "-> Same Angle selected." << std::endl;}}
}

void PrimaryGenerator::SetGunMode(G4int newMode)
{
    if (newMode == 0) 
    {   
        SpectraMode = 0; 
        if (threadID == 0) {std::cout << "-> Monocromatic Mode Selected" << std::endl;}
        energyHistogram.clear();
    }
    if (newMode == 1) 
    {
        SpectraMode = 1; 
        spectrumFile = "fSpectrum80.txt";
        if (threadID == 0) {std::cout << "-> Real 80kVp Spectrum Selected" << std::endl; std::cout << std::endl;}
        SpectraFunction();
        energyHistogram.clear();
    }
    if (newMode == 2) 
    {
        SpectraMode = 2; 
        spectrumFile = "fSpectrum140.txt";
        if (threadID == 0) {std::cout << "-> Real 140kVp Spectrum Selected" << std::endl; std::cout << std::endl;}
        SpectraFunction();
        energyHistogram.clear();
    }
}