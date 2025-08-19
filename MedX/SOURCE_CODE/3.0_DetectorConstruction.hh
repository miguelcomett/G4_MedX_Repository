#ifndef DetectorConstruction_hh
#define DetectorConstruction_hh

#include <vector> 
#include <string>
#include <filesystem>
#include <iostream>
#include <random>
#include <cmath>

#include "G4SystemOfUnits.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Box.hh"
#include "G4PVPlacement.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4NistManager.hh"
#include "G4LogicalVolume.hh"
#include "G4GenericMessenger.hh"
#include "G4UnionSolid.hh"
#include "G4VisAttributes.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4Material.hh"
#include "G4Tubs.hh"
#include "G4RandomTools.hh"
#include "G4RunManager.hh"
#include "G4SubtractionSolid.hh"
#include "G4VSolid.hh"
#include "G4Sphere.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4SubtractionSolid.hh"
#include "G4Ellipsoid.hh"
#include "G4MultiUnion.hh"
#include "G4UserLimits.hh"
#include "G4GeometryManager.hh"
#include "G4GeometryTolerance.hh"
#include "G4UnitsTable.hh"   

#include "3.1_DetectorAction.hh"
#include "3.2_Geometry3D.hh"
#include "3.3_GeometryReader.hh"

extern int arguments;

class DetectorConstruction:public G4VUserDetectorConstruction
{   
    public:

        DetectorConstruction();
        ~DetectorConstruction() override;

        void DefineMaterials();
        void ConstructSDandField() override;
        G4VPhysicalVolume * Construct() override;

        G4LogicalVolume * GetScoringVolume() const {return logicRadiator;}

        std::vector<G4LogicalVolume*> scoringVolumes;
        std::vector<G4LogicalVolume*> GetAllScoringVolumes() const {return scoringVolumes;}

        G4Material * GetMaterial() const {return materialTarget;}
	    G4double GetThickness() const {return targetThickness;}
        G4double GetThoraxAngle() const {return thoraxAngle;}

        G4bool  isArm, isHealthyBone, isOsteoBone, isBoneDivided, 
                is3DModel, isHeart, isLungs, isRibcage, isThorax, isTumorSTL, isTraquea,
                checkOverlaps, isTumorRandom, isTestParametrization, isFixed, isDebug;

        G4bool Getis3DModel() const {return is3DModel;}

        std::vector<G4double> GetTargetParameters() const {return {x_length, y_length, y_position, z_position, TargetAngle};}
    
    private:

        void ConstructTarget();
        void ConstructHealthyBone();
        void ConstructOsteoporoticBone();
        void ConstructArm();
        void ConstructTissue();
        void ConstructBoneDivided();
        void ConstructThorax();
        void ConstructTumor(int i);
        void ConstructEllipsoid(G4double aa, G4double bb, G4double cc, G4RotationMatrix* rot, G4ThreeVector EllipsoidPos, G4String name);
        void EllipsoidsParametrization();

        G4GenericMessenger * DetectorMessenger;

        std::string currentPath, modelPath;
        
        const G4double pi = 3.14159265358979323846;

        G4int DetectorColumnsCount, DetectorRowsCount, numPores, numTumores;
        
        G4double innerBoneRadius, outerBoneRadius, boneHeight, poreRadius, xWorld, yWorld, zWorld, 
                 regionMinZ, regionMaxZ, regionMinRadius, regionMaxRadius, r, theta, z, x, y, 
                 x_length, y_length, y_position, z_position, TargetAngle, ThoraxHalfHeight,
                 innerMuscleRadius, outerMuscleRadius, innerGrasaRadius, outerGrasaRadius, innerSkinRadius, outerSkinRadius,
                 fractionMass_VO2, fractionMass_SiO2, fTargetAngle, thoraxAngle, targetThickness, 
                 tumorRadius, a, b, c, angleX, angleY, angleZ, verify, randomNum, aRight, bRight, cRight, aLeft, bLeft, cLeft;

        G4Box    * solidWorld, * solidDetector, * solidRadiator;
        G4Tubs   * solidBone, * solidMuscle, * solidGrasa, * solidSkin, * solidBone2, * osteoBone, * healthyBone; 
        G4Sphere * pore,  * tumorSphere;
        G4VSolid * porousBone; 
        G4Ellipsoid * ellipsoidSolid;

        G4LogicalVolume   * logicWorld, * logicRadiator, * logicDetector, * logicHealthyBone, * logicOsteoBone, * logicMuscle, 
                          * logicGrasa, * logicSkin, * logicOs, * logicHealthy, 
                          * logicLungs, * logicHeart, * logic_Thorax_inner, * logic_Thorax_outer, * logicRibcage, * logicFiller, * logicTumor, * logicTraquea,
                          * logicTumorReal, * ellipsoidLogic;
        
        G4VPhysicalVolume * physicalWorld, * physicalRadiator, * physicalDetector, * physBone, * physArm, 
                          * physMuscle, * physGrasa, * physSkin, * physOs, * physHealthy;
                        
        G4ThreeVector samplePosition, DetectorPosition, porePosition, osteo_position, healthy_position, Radiator_Position, 
                      tumorPosition, correctedTumorPosition, selectedCenter, ellipsoidPosition1, ellipsoidPosition2, leftEllipsoidCenter, rightEllipsoidCenter;
                    
        G4RotationMatrix * TargetRotation, * armRotation, * Model3DRotation, * originMatrix, * elipsoidRot, * elipsoidRot2; 

        G4Element  * C, * Al, * N, * O, * Ca, * Mg, * V, * Cd, * Te, * W;
        G4Material * SiO2, * H2O, * Aerogel, * worldMaterial, * Calcium, * Magnesium, * Aluminum, * Air, * Vacuum, * Silicon, * materialTarget, 
                   * CadTel, * vanadiumGlassMix, * amorphousGlass, * Wolframium, * V2O5, 
                   * Adipose, * Skin, * Muscle, * Bone, * OsBone, * compactBone, * TissueMix, * Light_Adipose, * Muscle_Sucrose;
        
        G4VisAttributes * RadiatorAttributes;
        
        STLGeometryReader * stlReader;
        
        G4TessellatedSolid * Ribcage_STL, * Lungs_STL, * Heart_STL, * Tumor_STL, * Traquea_STL, * Thorax_outer_STL, * Thorax_inner_STL;
        
        G4SubtractionSolid * subtractedLungs, * subtractedThorax, * Thorax_outer, * subtractedHeart;

        //Distribuciones
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> randomDist;
        std::uniform_real_distribution<> radiusDist;
        std::uniform_real_distribution<> posXDist;
        std::uniform_real_distribution<> posYDist;
        std::uniform_real_distribution<> posZDist;
};

#endif 