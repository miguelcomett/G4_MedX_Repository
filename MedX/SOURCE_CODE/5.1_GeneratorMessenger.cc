#include "5.1_GeneratorMessenger.hh"

PrimaryGeneratorMessenger::PrimaryGeneratorMessenger(PrimaryGenerator * gun)
{
	fGun = gun;

	fPgunX = new G4UIcmdWithADoubleAndUnit("/Pgun/X", this);
	fPgunX -> SetGuidance("Set the source X position.");
	fPgunX -> SetParameterName("GunXpos", true);

	fPgunY = new G4UIcmdWithADoubleAndUnit("/Pgun/Y", this);
	fPgunY -> SetGuidance("Set the source Y position.");
	fPgunY -> SetParameterName("GunYpos", true);
	
	fPgunZ = new G4UIcmdWithADoubleAndUnit("/Pgun/Z", this);
	fPgunZ -> SetGuidance("Set the source Z position.");
	fPgunZ -> SetParameterName("GunZpos", true);

	fPgunXgauss = new G4UIcmdWithABool("/Pgun/gaussX", this);
	fPgunXgauss -> SetGuidance("Set the source X Distribution.");
	fPgunXgauss -> SetParameterName("GunGaussX", true);
	
	fPgunXcos = new G4UIcmdWithABool("/Pgun/Xcos", this);
	fPgunXcos -> SetGuidance("Set the source X span function.");
	fPgunXcos -> SetParameterName("GunXcos", true);
	
	fPgunSpanX = new G4UIcmdWithADouble("/Pgun/SpanX", this);
	fPgunSpanX -> SetGuidance("Set the source X length.");
	fPgunSpanX -> SetParameterName("GunSpanY", true);

	fPgunSpanY = new G4UIcmdWithADouble("/Pgun/SpanY", this);
	fPgunSpanY -> SetGuidance("Set the source Y length.");
	fPgunSpanY -> SetParameterName("GunSpanY", true);

	fPgunAngle = new G4UIcmdWithADouble("/Pgun/Angle", this);
	fPgunAngle -> SetGuidance("Set the source GunAngle.");
	fPgunAngle -> SetParameterName("GunAngle", true);
	// fPgunAngle->SetDefaultUnit("deg");

	fSpectraMode = new G4UIcmdWithAnInteger("/Pgun/Mode", this); 
	fSpectraMode -> SetGuidance("Set the particle GunMode");
	fSpectraMode -> SetGuidance("0: monocromatic energy");
	fSpectraMode -> SetGuidance("1: 80kVp real custom spectrum"); 
	fSpectraMode -> SetGuidance("2: 140kVp real custom spectrum"); 
	fSpectraMode -> SetParameterName("GunMode", true);
}

PrimaryGeneratorMessenger::~PrimaryGeneratorMessenger()
{
	delete fPgunX; delete fPgunY; delete fPgunZ;
	delete fPgunXgauss; delete fPgunXcos;
	delete fPgunSpanX; delete fPgunSpanY;
	delete fPgunAngle; 
	delete fSpectraMode; 
}

void PrimaryGeneratorMessenger::SetNewValue(G4UIcommand * command, G4String newValue)
{
	if (command == fPgunX)
	{
	    G4double GunXpos = fPgunX -> GetNewDoubleValue(newValue);
	    fGun -> SetGunXpos(GunXpos);
	}

	if (command == fPgunY)
	{
	    G4double GunYpos = fPgunY -> GetNewDoubleValue(newValue); 
		fGun -> SetGunYpos(GunYpos);
	}

	if (command == fPgunZ)
	{
	    G4double GunZpos = fPgunZ -> GetNewDoubleValue(newValue);
	    fGun -> SetGunZpos(GunZpos);
	}

	if (command == fPgunXcos)
	{
		G4bool GunXtriangular = fPgunXcos -> GetNewBoolValue(newValue);
	    fGun -> SetGunXcos(GunXtriangular);
	}

	if (command == fPgunXgauss)
	{
		G4bool GunXgauss = fPgunXgauss -> GetNewBoolValue(newValue);
	    fGun -> SetGunXGauss(GunXgauss);
	}

	if (command == fPgunSpanX)
	{
		G4double GunSpanX = fPgunSpanX -> GetNewDoubleValue(newValue);
	    fGun -> SetGunSpanX(GunSpanX);
	}

	if (command == fPgunSpanY)
	{
		G4double GunSpanY = fPgunSpanY -> GetNewDoubleValue(newValue);
	    fGun -> SetGunSpanY(GunSpanY);
	}
	
	if (command == fPgunAngle)
	{
	    G4double GunAngle = fPgunAngle -> GetNewDoubleValue(newValue);
	    fGun -> SetGunAngle(GunAngle);
	}

	if (command == fSpectraMode)
	{
	    G4int GunMode = fSpectraMode -> GetNewIntValue(newValue); 
	    fGun -> SetGunMode(GunMode);
	}
}