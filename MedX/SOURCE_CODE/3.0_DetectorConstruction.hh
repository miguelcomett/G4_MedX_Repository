#ifndef DetectorConstruction_hh
#define DetectorConstruction_hh

#include <vector> 
#include <string>
#include <filesystem>
#include <iostream>
#include <random>

#include "G4SystemOfUnits.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Box.hh"
#include "G4PVPlacement.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4NistManager.hh"
#include "G4LogicalVolume.hh"
#include "G4GenericMessenger.hh"
#include "G4UnionSolid.hh"
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

        G4LogicalVolume * GetScoringVolume() const {return scoringVolume_0;}

        std::vector<G4LogicalVolume*> scoringVolumes;
        std::vector<G4LogicalVolume*> GetAllScoringVolumes() const {return scoringVolumes;}

        G4Material * GetMaterial() const {return materialTarget;}
	    G4double GetThickness() const {return targetThickness;}
        G4double GetThoraxAngle() const {return thoraxAngle;}

        G4bool  isArm, isHealthyBone, isOsteoBone, isBoneDivided, 
                is3DModel, isHeart, isLungs, isRibcage, isFiller, isThorax, isTumorReal, isTraquea,
                checkOverlaps, isTumorRandom, isTestParametrization, isFixed, isDebug;

        G4bool Getis3DModel() const {return is3DModel;}
    
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

        G4int DetectorColumnsCount, DetectorRowsCount, numPores, numTumores;
        
        G4double innerBoneRadius, outerBoneRadius, boneHeight, poreRadius, xWorld, yWorld, zWorld, 
                 regionMinZ, regionMaxZ, regionMinRadius, regionMaxRadius, r, theta, z, x, y,
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
                          * logicLungs, * logicHeart, * logicThorax, * logicRibcage, * logicFiller, * logicTumor, * logicTraquea,
                          * scoringVolume_0, * scoringVolume_1, * scoringVolume_2, * scoringVolume_3, 
                          * scoringVolume_4, * scoringVolume_5, * scoringVolume_6, * scoringVolume_7, * scoringVolume_8, * logicTumorReal, * ellipsoidLogic;
        G4VPhysicalVolume * physicalWorld, * physicalRadiator, * physicalDetector, * physBone, * physArm, 
                          * physMuscle, * physGrasa, * physSkin, * physOs, * physHealthy;
                        
        G4ThreeVector samplePosition, DetectorPosition, porePosition, osteo_position, healthy_position, Radiator_Position, 
                      tumorPosition, correctedTumorPosition, selectedCenter, ellipsoidPosition1, ellipsoidPosition2, leftEllipsoidCenter, rightEllipsoidCenter;
                    
        G4RotationMatrix * armRotation, * Model3DRotation, * originMatrix, * elipsoidRot, * elipsoidRot2; 

        G4Element  * C, * Al, * N, * O, * Ca, * Mg, * V, * Cd, * Te, * W;
        G4Material * SiO2, * H2O, * Aerogel, * worldMaterial, * Calcium, * Magnesium, * Aluminum, * Air, * Silicon, * materialTarget, 
                   * CadTel, * vanadiumGlassMix, * amorphousGlass, * Wolframium, * V2O5, 
                   * Adipose, * Skin, * Muscle, * Bone, * OsBone, * compactBone, * TissueMix, * Light_Adipose, * Muscle_Sucrose;
        
        STLGeometryReader * stlReader;
        G4TessellatedSolid * Ribcage, * Lungs, * Heart, * TumorReal, * Traquea, * Tumor;
        G4VSolid * Thorax1, * Thorax2, * AccumulatedLungs;
        G4SubtractionSolid * subtractedLungs_1, * subtractedLungs_2, 
                           * subtractedThorax_1, * subtractedThorax_2, * subtractedThorax_3,
                           * subtractedFiller_1, * subtractedFiller_2, * subtractedFiller_3, 
                           * subtractedFiller_4, * subtractedHeart_1;

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