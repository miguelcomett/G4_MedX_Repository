#ifndef PRIMARY_GENERATOR_MESSENGER_H
#define PRIMARY_GENERATOR_MESSENGER_H

#include "G4UImessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"

#include "G4SystemOfUnits.hh"
#include "G4UImanager.hh"
#include "G4RunManager.hh"

#include "5.0_PrimaryGenerator.hh"

class PrimaryGenerator; 

class PrimaryGeneratorMessenger : public G4UImessenger
{
	public:

		PrimaryGeneratorMessenger(PrimaryGenerator * gun);
		~PrimaryGeneratorMessenger() override;
		void SetNewValue(G4UIcommand * command, G4String newValue) override;
    	
	private:
		
		PrimaryGenerator * fGun;

		G4int threadID;
		
		G4UIcmdWithABool * fPgunXcos, * fPgunXgauss;   
		G4UIcmdWithAnInteger * fSpectraMode;
		G4UIcmdWithADouble * fPgunSpanX, * fPgunSpanY, * fPgunAngle;
		G4UIcmdWithADoubleAndUnit * fPgunX, * fPgunY, * fPgunZ; 
};

#endif
