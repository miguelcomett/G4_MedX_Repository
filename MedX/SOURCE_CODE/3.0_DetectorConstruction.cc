#include "3.0_DetectorConstruction.hh"

DetectorConstruction::DetectorConstruction()
{
    G4GeometryManager::GetInstance() -> SetWorldMaximumExtent(100.0 * cm);

    DefineMaterials();
    stlReader = new STLGeometryReader();

    DetectorMessenger = new G4GenericMessenger(this, "/myDetector/", "Detector Construction");
    DetectorMessenger -> DeclareProperty("nColumns", DetectorColumnsCount, "Number of columns");
    DetectorMessenger -> DeclareProperty("nRows", DetectorRowsCount, "Number of rows");
    DetectorMessenger -> DeclareProperty("ThicknessTarget", targetThickness, "Thickness of the target");
    DetectorMessenger -> DeclareProperty("Rotation", thoraxAngle, "Rotate the 3D model");

    DetectorColumnsCount = 1;
    DetectorRowsCount = 1;

    targetThickness = 12 * mm;

    samplePosition = G4ThreeVector(0.0, 0.0, 0.0);

    boneHeight = 60 * mm;
    innerBoneRadius = 0.0;
    outerBoneRadius = 22.5 * mm;
    armRotation = new G4RotationMatrix(0, 90*deg, 0);

    thoraxAngle = 0;
    
    ellipsoidPosition1 = G4ThreeVector(88.0 * mm, 5.0 * mm, -8.0 * mm); if (isDebug) {G4cout << "Posición rotada pulmón derecho: " << ellipsoidPosition1 << G4endl;}
    ellipsoidPosition2 = G4ThreeVector(-93 * mm, 5.0 * mm, -13.0 * mm); if (isDebug) {G4cout << "Posición rotada pulmón izquierdo: " << ellipsoidPosition2 << G4endl;}
    
    gen = std::mt19937(rd());
    randomDist = std::uniform_real_distribution<>( 0.0, 1.0);
    radiusDist = std::uniform_real_distribution<>( 05.0 * mm, 20.0 * mm);
    posXDist   = std::uniform_real_distribution<>(-01.0 * mm, 01.0 * mm);
    posYDist   = std::uniform_real_distribution<>(-40.0 * mm, 40.0 * mm);
    posXDist   = std::uniform_real_distribution<>(-01.0 * mm, 01.0 * mm);

    isArm = false;
        isBoneDivided = false;
        isHealthyBone = true;
        isOsteoBone = false;
    is3DModel = false;
        isHeart = true;
        isLungs = true;
            isTraquea = false;
            isTumorReal = true;
        isRibcage = true;
        isThorax = true;
        isFiller = true;
        
        isTumorRandom = false;
            isFixed = true; //Default random false
            isTestParametrization = false;
            isDebug = false;
}

DetectorConstruction::~DetectorConstruction() {delete DetectorMessenger; delete stlReader;}

G4VPhysicalVolume * DetectorConstruction::Construct()
{
    xWorld = 0.5*m, yWorld = 0.5*m, zWorld = 0.5*m;
    solidWorld = new G4Box("SolidWorld", xWorld, yWorld, zWorld);
    logicWorld = new G4LogicalVolume(solidWorld, worldMaterial, "LogicalWorld");
    physicalWorld = new G4PVPlacement(0, G4ThreeVector(0.0, 0.0, 0.0), logicWorld, "PhysicalWorld", 0, false, 0, true);

    if (arguments == 3) {ConstructTarget();} else 
    if (isArm) ConstructArm(); else if (is3DModel) ConstructThorax();

    if (isTestParametrization) //Test ellipsoids where the tumor will reside 
    {
        //                  a,  b,  c,         rotationMat                                VectorPos
        ConstructEllipsoid(100, 25, 40, new G4RotationMatrix(100*deg,  0*deg, 5*deg), ellipsoidPosition1, "LeftLung");
        ConstructEllipsoid(100, 25, 40, new G4RotationMatrix(-100*deg, 0*deg, 0*deg), ellipsoidPosition2, "RightLung");
    }

    solidDetector = new G4Box("solidDetector", xWorld/DetectorRowsCount, yWorld/DetectorColumnsCount, 0.01*m);
    logicDetector = new G4LogicalVolume(solidDetector, Silicon, "logicalDetector");
    checkOverlaps = false;
    for(G4int i = 0; i < DetectorRowsCount; i++) for (G4int j = 0; j < DetectorColumnsCount; j++)
    {
        DetectorPosition = G4ThreeVector(-0.5*m + (i+0.5)*m/DetectorRowsCount, -0.5*m + (j+0.5)*m/DetectorColumnsCount, 0.49*m);
        physicalDetector = new G4PVPlacement(0, DetectorPosition, logicDetector, "physicalDetector", logicWorld, false, j+(i*DetectorColumnsCount), checkOverlaps);
    }

    return physicalWorld;
}

void DetectorConstruction::ConstructSDandField()
{
    SensitiveDetector * sensitiveDetector = new SensitiveDetector("sensitiveDetector");
    logicDetector -> SetSensitiveDetector(sensitiveDetector);
}

// Build Geometries ===============================================================================================================================

void DetectorConstruction::ConstructTarget()
{ 
    materialTarget = Aluminum;

    Radiator_Position = G4ThreeVector(0.0, 0.0, 0.25*m);

    solidRadiator = new G4Box("solidRadiator", 0.45*m, 0.45*m, targetThickness/2);
    logicRadiator = new G4LogicalVolume(solidRadiator, materialTarget, "logicalRadiator");
    physicalRadiator = new G4PVPlacement(0, Radiator_Position, logicRadiator, "PhysicalRadiator", logicWorld, false, 0, true);

    scoringVolume_0 = logicRadiator;
}

void DetectorConstruction::ConstructArm() 
{
    innerMuscleRadius = outerBoneRadius;
    outerMuscleRadius = innerMuscleRadius + 25 * mm;
    innerGrasaRadius  = outerMuscleRadius;
    outerGrasaRadius  = innerGrasaRadius + 5 * mm;
    innerSkinRadius   = outerGrasaRadius;
    outerSkinRadius   = innerSkinRadius + 1.5 * mm;

    if (isBoneDivided) {ConstructBoneDivided();} else if (isHealthyBone) {ConstructHealthyBone();} else if (isOsteoBone) {ConstructOsteoporoticBone();}

    solidMuscle = new G4Tubs("Muscle",  innerMuscleRadius, outerMuscleRadius, boneHeight/2, 0.0, 360.0*deg);
    solidGrasa  = new G4Tubs("Grasa",   innerGrasaRadius, outerGrasaRadius,   boneHeight/2, 0.0, 360.0*deg);
    solidSkin   = new G4Tubs("Skin",    innerSkinRadius, outerSkinRadius,     boneHeight/2, 0.0, 360.0*deg);

    logicMuscle = new G4LogicalVolume(solidMuscle, Muscle, "LogicMuscle");
    logicGrasa = new G4LogicalVolume(solidGrasa, Adipose, "LogicGrasa");
    logicSkin = new G4LogicalVolume(solidSkin, Skin, "LogicSkin");

    physMuscle = new G4PVPlacement(armRotation, samplePosition, logicMuscle, "physMuscle", logicWorld, false, 0, true);
    physGrasa = new G4PVPlacement(armRotation, samplePosition, logicGrasa, "physGrasa", logicWorld, false, 0, true);
    physSkin = new G4PVPlacement(armRotation, samplePosition, logicSkin, "physSkin", logicWorld, false, 0, true);

    scoringVolume_3 = logicMuscle;
    scoringVolume_4 = logicSkin;
    scoringVolume_5 = logicGrasa;

    scoringVolumes.push_back(scoringVolume_3);
    scoringVolumes.push_back(scoringVolume_4);
    scoringVolumes.push_back(scoringVolume_5);
}

void DetectorConstruction::ConstructHealthyBone() 
{
    solidBone = new G4Tubs("Bone", innerBoneRadius, outerBoneRadius, boneHeight/2, 0.0, 360.0*deg);
    logicHealthyBone = new G4LogicalVolume(solidBone, Bone, "LogicBone");
    physBone = new G4PVPlacement(armRotation, samplePosition, logicHealthyBone, "physBone", logicWorld, false, 0, true);

    scoringVolume_1 = logicHealthyBone;
    scoringVolumes.push_back(scoringVolume_1);
}

void DetectorConstruction::ConstructOsteoporoticBone() 
{   
    solidBone = new G4Tubs("Bone", innerBoneRadius, outerBoneRadius, boneHeight/2, 0.0, 360.0*deg);
    porousBone = solidBone;

    numPores = 20;
    poreRadius = 0.30 * cm;

    pore = new G4Sphere("Pore", 0, poreRadius, 0 * deg, 360 * deg, 0 * deg, 180 * deg);

    regionMinZ = 0; 
    regionMaxZ = boneHeight / 2; 
    regionMinRadius = 0; 
    regionMaxRadius = outerBoneRadius;

    for (int i = 1; i <= numPores; i++)
    {
        r      = G4RandGauss::shoot(0.6, 0.25) * (regionMaxRadius - regionMinRadius);
        theta  = G4UniformRand() * 360.0 * deg;
        z      = (G4UniformRand() * (regionMaxZ - regionMinZ - poreRadius)) + regionMinZ;
      
        x = r * std::cos(theta);
        y = r * std::sin(theta);

        porePosition = G4ThreeVector(x, y, -z);
        porousBone = new G4SubtractionSolid("PorousBone", porousBone, pore, 0, porePosition);
    }

    logicOsteoBone = new G4LogicalVolume(porousBone, Bone, "PorousBoneLogical");
    physBone = new G4PVPlacement(armRotation, samplePosition, logicOsteoBone, "physBone", logicWorld, false, 0);

    scoringVolume_1 = logicOsteoBone;
    scoringVolumes.push_back(scoringVolume_1);
}

void DetectorConstruction::ConstructBoneDivided()
{
    osteoBone = new G4Tubs("Healty_Bone", innerBoneRadius, outerBoneRadius, boneHeight/4, 0.0, 360.0*deg);
    healthyBone = new G4Tubs("Osteo_Bone",  innerBoneRadius, outerBoneRadius, boneHeight/4, 0.0, 360.0*deg);
    
    osteo_position = G4ThreeVector(0, boneHeight/4, 0);
    logicOs = new G4LogicalVolume(osteoBone, OsBone, "LogicOs");
    physOs  = new G4PVPlacement(armRotation, osteo_position, logicOs, "physOs", logicWorld, false, 0, true);

    healthy_position = G4ThreeVector(0, -boneHeight/4, 0);
    logicHealthy = new G4LogicalVolume(healthyBone, Bone, "LogiHealthy");
    physHealthy  = new G4PVPlacement(armRotation, healthy_position, logicHealthy, "physHealthy", logicWorld, false, 0, true);

    scoringVolume_1 = logicOs;
    scoringVolume_2 = logicHealthy;
    scoringVolumes.push_back(scoringVolume_1);
    scoringVolumes.push_back(scoringVolume_2);
}

// Load 3D Models ===============================================================================================================================

void DetectorConstruction::ConstructThorax()
{
    angleX = 0 * deg; angleY = -90 * deg; angleZ = (thoraxAngle + 180) * deg; 
    Model3DRotation = new G4RotationMatrix(angleX, angleY, angleZ);

    G4STL stl; 
    
    #ifdef __APPLE__
        std::string currentPath = std::filesystem::current_path().string();
        std::string modelPath = std::filesystem::path(currentPath).parent_path().string() + "/3D_Models/";
    #else
        // Obtener el directorio actual (donde está el ejecutable, probablemente en "build/Release")
        std::string currentPath = std::filesystem::current_path().string();

        // Navegar al directorio superior y luego a "ROOT"
        std::string modelPath = std::filesystem::path(currentPath).parent_path().parent_path().string() + "\\3D_Models\\";
        //std::string modelPath = "C:\\Users\\conej\\Documents\\Universidad\\Geant4\\Projects\\Models2\\"; // Define el directorio de los modelos 3D
    #endif

    originMatrix = new G4RotationMatrix(0, 0, 0);
        
    if (arguments == 1) {G4cout << G4endl;}
    G4cout << "=============== 3D MODELS ====================================" << G4endl; 
    G4cout << "-> Model Rotation about Z: " << thoraxAngle << "°" << G4endl;

    Heart = stl.Read(modelPath + "HEART3.stl");
    if (Heart && isHeart)
    {
        if (isTraquea)
        {
            subtractedHeart_1 = new G4SubtractionSolid("Heart1", Heart, Traquea, originMatrix, samplePosition);
            logicHeart = new G4LogicalVolume(subtractedHeart_1, Muscle, "Heart");
            new G4PVPlacement(Model3DRotation, samplePosition, logicHeart, "Heart", logicWorld, false, 0, true);
        }
        else
        {
            logicHeart = new G4LogicalVolume(Heart, Muscle, "Heart");
            new G4PVPlacement(Model3DRotation, samplePosition, logicHeart, "Heart", logicWorld, false, 0, true);
        }

        scoringVolume_1 = logicHeart;
        scoringVolumes.push_back(scoringVolume_1);

        G4cout << "> HEART model imported succesfully" << G4endl;
    }
    if (isHeart && !Heart)
    {
        G4cout << "--> HEART model not found" << G4endl;
        G4cout << "Critical error: Stopping the Simulation" << G4endl;
        G4cout << "==============================================================" << G4endl; G4cout << G4endl;
        std::exit(EXIT_FAILURE);
    }

    TumorReal = stl.Read(modelPath + "TUMOR.stl");
    if (TumorReal && isTumorReal)
    {
        logicTumorReal = new G4LogicalVolume(TumorReal, Muscle_Sucrose, "TumorReal");
        new G4PVPlacement(Model3DRotation, samplePosition, logicTumorReal, "TumorReal", logicWorld, false, 0, true);

        scoringVolume_6 = logicTumorReal;
        scoringVolumes.push_back(scoringVolume_6);

        G4cout << "> Tumor model imported succesfully" << G4endl;
    }
    if (isTumorReal && !TumorReal)
    {
        G4cout << "--> Tumor model not found" << G4endl;
        G4cout << "Critical error: Stopping the Simulation" << G4endl;
        G4cout << "==============================================================" << G4endl; G4cout << G4endl;
        std::exit(EXIT_FAILURE);
    }

    Traquea = stl.Read(modelPath + "thyroid.stl");
    if (Traquea && isTraquea)
    {
        logicTraquea = new G4LogicalVolume(Traquea, Air, "Traquea");
        new G4PVPlacement(Model3DRotation, samplePosition, logicTraquea, "Traquea", logicWorld, false, 0, true);

        scoringVolume_7 = logicTraquea;
        scoringVolumes.push_back(scoringVolume_7);

        G4cout << "> Traquea model imported succesfully" << G4endl;
    }
    if (isTraquea && !Traquea)
    {
        G4cout << "--> Traquea model not found" << G4endl;
        G4cout << "Critical error: Stopping the Simulation" << G4endl;
        G4cout << "==============================================================" << G4endl; G4cout << G4endl;
        std::exit(EXIT_FAILURE);
    }

    Lungs = stl.Read(modelPath + "LUNGS1.stl");
    if (Lungs && isLungs)
    {
        if (isTumorRandom)
        {
            AccumulatedLungs = Lungs; // Se puede igualar un solido a un tesselate
            numTumores = 2; 

            for (int i = 1; i <= numTumores; i++)
            {
                ConstructTumor(i); 
                correctedTumorPosition = G4ThreeVector(-tumorPosition.x(), tumorPosition.z(), tumorPosition.y());
                AccumulatedLungs = new G4SubtractionSolid("LungsWithTumorHole", AccumulatedLungs, tumorSphere, Model3DRotation, correctedTumorPosition);
            }

            logicLungs = new G4LogicalVolume(AccumulatedLungs, Air, "Lungs"); // Crear el volumen lógico del sólido resultante
            new G4PVPlacement(Model3DRotation, samplePosition, logicLungs, "Lungs", logicWorld, false, 0, true);
        }

        if (isTumorReal)
        {
            subtractedLungs_1 = new G4SubtractionSolid("Lung0", Lungs, TumorReal, originMatrix, samplePosition);
            logicLungs = new G4LogicalVolume(subtractedLungs_1, Air, "Lungs");
        }
        if (isTraquea)
        {
            subtractedLungs_2 = new G4SubtractionSolid("Lung1", Lungs, Traquea, originMatrix, samplePosition);
            logicLungs = new G4LogicalVolume(subtractedLungs_2, Air, "Lungs");
        }
        if (isTumorReal && isTraquea)
        {
            subtractedLungs_1 = new G4SubtractionSolid("Lung0", Lungs, TumorReal, originMatrix, samplePosition);
            subtractedLungs_2 = new G4SubtractionSolid("Lung1", subtractedLungs_1, Traquea, originMatrix, samplePosition);
            logicLungs = new G4LogicalVolume(subtractedLungs_2, Air, "Lungs");
        }
        if (!isTumorReal && !isTraquea){logicLungs = new G4LogicalVolume(Lungs, Air, "Lungs");}
        
        new G4PVPlacement(Model3DRotation, samplePosition, logicLungs, "Lungs", logicWorld, false, 0, true);
        scoringVolume_2 = logicLungs;
        scoringVolumes.push_back(scoringVolume_2);
       
        G4cout << "> LUNGS model imported succesfully" << G4endl;
    }
    if (isLungs && !Lungs) 
    {
        G4cout << "--> LUNGS model not found" << G4endl;
        G4cout << "Critical error: Stopping the Simulation" << G4endl;
        G4cout << "==============================================================" << G4endl; G4cout << G4endl;
        std::exit(EXIT_FAILURE);
    }

    Ribcage = stl.Read(modelPath + "RIBCAGE_Real.stl");
    if (Ribcage && isRibcage) 
    {
        logicRibcage = new G4LogicalVolume(Ribcage, Bone, "Ribcage");
        new G4PVPlacement(Model3DRotation, samplePosition, logicRibcage, "Ribcage", logicWorld, false, 0, true);
        
        scoringVolume_3 = logicRibcage;
        scoringVolumes.push_back(scoringVolume_3);
        
        G4cout << "> RIBCAGE model imported succesfully" << G4endl;
    }
    if (isRibcage && !Ribcage) 
    {
        G4cout << "--> RIBCAGE model not found" << G4endl;
        G4cout << "Critical error: Stopping the Simulation" << G4endl;
        G4cout << "==============================================================" << G4endl; G4cout << G4endl;
        std::exit(EXIT_FAILURE);
    }

    Thorax1 = stl.Read(modelPath + "TORAX_Real.stl");
    Thorax2 = stl.Read(modelPath + "TORAX_Real0.stl");
    if (Thorax1 && Thorax2 && isThorax) 
    {
        if (isTraquea)
        {
            subtractedThorax_1 = new G4SubtractionSolid("ThoraxTraquea", Thorax1, Traquea, originMatrix, samplePosition);
            subtractedThorax_2 = new G4SubtractionSolid("SoftWithBoneHole", subtractedThorax_1, Ribcage, originMatrix, samplePosition); 
            subtractedThorax_3 = new G4SubtractionSolid("SoftWithBoneAndToraxHole", subtractedThorax_2, Thorax2, originMatrix, samplePosition); 
            logicThorax = new G4LogicalVolume(subtractedThorax_3, TissueMix, "Thorax"); 
        }
        else
        {
            subtractedThorax_1 = new G4SubtractionSolid("SoftWithBoneHole", Thorax1, Ribcage, originMatrix, samplePosition); 
            subtractedThorax_2 = new G4SubtractionSolid("SoftWithBoneAndToraxHole", subtractedThorax_1, Thorax2, originMatrix, samplePosition);
            logicThorax = new G4LogicalVolume(subtractedThorax_2, TissueMix, "Thorax");
        }
       
        new G4PVPlacement(Model3DRotation, G4ThreeVector(0, 0, 0), logicThorax, "Thorax", logicWorld, false, 0, true);
        scoringVolume_4 = logicThorax;
        scoringVolumes.push_back(scoringVolume_4);
       
        G4cout << "> THORAX model imported succesfully" << G4endl;
    }
    if (isThorax && (!Thorax1 || !Thorax2) ) 
    {
        G4cout << "--> THORAX model not found" << G4endl;
        G4cout << "Critical error: Stopping the Simulation" << G4endl;
        G4cout << "==============================================================" << G4endl; G4cout << G4endl;
        std::exit(EXIT_FAILURE);
    }

    if (isFiller)
    {
        subtractedFiller_1 = new G4SubtractionSolid("Inner0", Thorax2, Lungs, originMatrix, samplePosition);
        subtractedFiller_2 = new G4SubtractionSolid("Inner1", subtractedFiller_1, Heart, originMatrix, samplePosition);
        subtractedFiller_3 = new G4SubtractionSolid("Inner2", subtractedFiller_2, Ribcage, originMatrix, samplePosition);

        if (isTraquea)
        {
            subtractedFiller_4 = new G4SubtractionSolid("Inner6", subtractedFiller_3, Traquea, originMatrix, samplePosition);
            logicFiller = new G4LogicalVolume(subtractedFiller_4, Light_Adipose, "Filler");
        }
        else {logicFiller = new G4LogicalVolume(subtractedFiller_3, Light_Adipose, "Filler");}
        
        new G4PVPlacement(Model3DRotation, G4ThreeVector(0, 0, 0), logicFiller, "Filler", logicWorld, false, 0, true);
        scoringVolume_5 = logicFiller;
        scoringVolumes.push_back(scoringVolume_5);
        
        G4cout << "> FILLER model imported succesfully" << G4endl;
    }
    if (isFiller && (!Heart || !Lungs || !Ribcage || !Thorax1 || !Thorax2 || !TumorReal || !Traquea) )
    {
        G4cout << "--> FILLER model not found" << G4endl;
        G4cout << "Critical error: Stopping the Simulation" << G4endl;
        G4cout << "==============================================================" << G4endl; 
        std::exit(EXIT_FAILURE);
    }

    G4cout << "==============================================================" << G4endl; G4cout << G4endl; 
}

// Create Tumor ===================================================================================================================================

void DetectorConstruction::ConstructTumor(int i)
{
    EllipsoidsParametrization();
    tumorRadius = radiusDist(gen);
    
    if (!isFixed)
    {
        while (true)
        {
            // Generar coordenadas aleatorias dentro del elipsoide // Reducir semiejes para incluir el radio
            x = (2.0 * posXDist(gen) - 1.0) * (a - tumorRadius); 
            y = (2.0 * posYDist(gen) - 1.0) * (b - tumorRadius);
            z = (2.0 * posZDist(gen) - 1.0) * (c - tumorRadius);

            verify = (x * x) / (a * a) + (y * y) / (b * b) + (z * z) / (c * c); 
            
            if (verify <= 1.0) // Verificar si el centro del tumor está dentro del elipsoide con suficiente espacio para el radio
            {
                tumorPosition = selectedCenter + G4ThreeVector(x, y, z);
                if (isDebug) {G4cout << "Ratio: " << verify << G4endl; G4cout << "Coordenadas sumadas-> x: " << x << " y: " << y << " z: " << z << G4endl;}
                break;
            }
        }
    }

    if (isFixed)
    {
        if (i == 1)
        {
            tumorRadius = 14.6276 * mm;
            tumorPosition = G4ThreeVector(89.7214,59.3028,1.60989);
        }
        else
        {
            tumorPosition = G4ThreeVector(69.6701,-23.7793,-14.1257);
            tumorRadius = 11.704 * mm; 
        }
    }
    
    tumorSphere = new G4Sphere("Tumor", 0, tumorRadius, 0*deg, 360*deg, 0*deg, 180*deg);
    logicTumor = new G4LogicalVolume(tumorSphere, Muscle_Sucrose, "Tumor");
    new G4PVPlacement(Model3DRotation, tumorPosition, logicTumor, "Tumor", logicWorld, false, 0, true);
    
    if (isDebug) {G4cout << "Radio del tumor generado: " << tumorRadius / mm << " mm" << G4endl; G4cout << "Tumor generado en posición: " << tumorPosition << G4endl;}
}

void DetectorConstruction::ConstructEllipsoid(G4double aa, G4double bb, G4double cc, G4RotationMatrix * rot, G4ThreeVector EllipsoidPos, G4String name)
{
    a = aa * mm; b = bb * mm; c = cc * mm; // Semiejes

    ellipsoidSolid = new G4Ellipsoid(name, a, b, c); elipsoidRot2 = rot;
    ellipsoidLogic = new G4LogicalVolume(ellipsoidSolid, Air, name);
    new G4PVPlacement(elipsoidRot2, EllipsoidPos, ellipsoidLogic, name, logicWorld, false, 0, true);

    if (isDebug) {G4cout << "> " << name << " creado con semiejes : " << "a = " << a / mm << " mm, " << "b = " << b / mm << " mm, " << "c = " << c / mm << " mm " << "en posición " << EllipsoidPos << G4endl;}
}

void DetectorConstruction::EllipsoidsParametrization() // Una vez con los parámetros se crea la region para el tumor
{
    leftEllipsoidCenter =  ellipsoidPosition1; // Centro del elipsoide izquierdo
    aLeft = 100.0 * mm; bLeft = 25.0 * mm; cLeft = 40.0 * mm; // Semiejes

    rightEllipsoidCenter =  ellipsoidPosition2; // Centro del elipsoide derecho
    aRight = 100.0 * mm; bRight = 25.0 * mm; cRight = 40.0 * mm; // Semiejes

    randomNum = randomDist(gen); // Generar un número aleatorio entre 0 y 1
    if (isDebug) {G4cout << "Número aleatorio: " << randomNum << G4endl;}

    if (randomNum < 0.5) // 0 para el elipsoide izquierdo
    {
        selectedCenter = leftEllipsoidCenter;
        a = bLeft; b = aLeft; c = cLeft;
        if (isDebug) {G4cout << "Tumor generado en el elipsoide izquierdo" << G4endl;}
    }
    else // 1 para el elipsoide derecho
    {
        selectedCenter = rightEllipsoidCenter;
        a = bRight; b = aRight; c = cRight;
        if (isDebug) {G4cout << "Tumor generado en el elipsoide derecho" << G4endl;}
    }
}

// Define materials =============================================================================================================================

void DetectorConstruction::DefineMaterials()
{
    G4NistManager * nist = G4NistManager::Instance();

    // Elements ========================================================================================

    C  = new G4Element("Carbon",     "C",  6,   12.01*g/mole);
    N  = new G4Element("Nitrogen",   "N",  7,   14.01*g/mole);
    O  = new G4Element("Oxygen",     "O",  8,   16.00*g/mole);
    Mg = new G4Element("Magnesium",  "Mg", 12,  24.31*g/mole);
    Ca = new G4Element("Calcium",    "Ca", 20,  40.08*g/mole);
    V  = new G4Element("Vanadium",   "V",  23,  50.94*g/mole);
    Cd = new G4Element("Cadmium",    "Cd", 48, 112.41*g/mole);
    Te = new G4Element("Tellurium",  "Te", 52, 127.60*g/mole);
    W  = new G4Element("Wolframium", "W",  74, 183.84*g/mole);

    Calcium = new G4Material("Calcium", 1.55*g/cm3, 1);
    Calcium -> AddElement(nist -> FindOrBuildElement("Ca"), 1);

    Magnesium = new G4Material("Magnesium", 1.74*g/cm3, 1);
    Magnesium -> AddElement(nist -> FindOrBuildElement("Mg"), 1);

    Aluminum = new G4Material("Aluminum", 2.70*g/cm3, 1);
    Aluminum -> AddElement(nist -> FindOrBuildElement("Al"), 1);

    Silicon = new G4Material("Silicon", 2.33*g/cm3, 1);
    Silicon -> AddElement(nist -> FindOrBuildElement("Si"), 1);

    Wolframium = new G4Material("Wolframium", 19.25*g/cm3, 1);
    Wolframium -> AddElement(nist -> FindOrBuildElement("W"), 1);

    // Compounds =======================================================================================

    worldMaterial = nist -> FindOrBuildMaterial("G4_AIR");
    
    Air = new G4Material("Air", 0.0001*g/cm3, 2);
    Air -> AddElement(N, 0.78);
    Air -> AddElement(O, 0.22);

    SiO2 = new G4Material("SiO2", 2.201*g/cm3, 2); 
    SiO2 -> AddElement(nist -> FindOrBuildElement("Si"), 1);
    SiO2 -> AddElement(nist -> FindOrBuildElement("O"), 2);

    H2O = new G4Material("H2O", 1.0*g/cm3, 2); 
    H2O -> AddElement(nist -> FindOrBuildElement("H"), 2);
    H2O -> AddElement(nist -> FindOrBuildElement("O"), 1);

    // Glass Materials =======================================================================================
    
    CadTel = nist -> FindOrBuildMaterial("G4_CADMIUM_TELLURIDE");

    V2O5 = new G4Material("V2O5", 3.36*g/cm3, 2);
    V2O5 -> AddElement(V, 2);
    V2O5 -> AddElement(O, 5);

    amorphousGlass = new G4Material("AmorphousGlass", 2.5*g/cm3, 2);
    amorphousGlass -> AddElement(nist -> FindOrBuildElement("Si"), 1);
    amorphousGlass -> AddElement(nist -> FindOrBuildElement("O"), 2);

    fractionMass_VO2 = 0.05, fractionMass_SiO2 = 1.0 - fractionMass_VO2;
    vanadiumGlassMix = new G4Material("VanadiumGlassMix", 2.7*g/cm3, 2);
    vanadiumGlassMix -> AddMaterial(V2O5, fractionMass_VO2);
    vanadiumGlassMix -> AddMaterial(amorphousGlass, fractionMass_SiO2);

    // Organic Materials =======================================================================================

    Bone = nist -> FindOrBuildMaterial("G4_B-100_BONE"); 
    compactBone = nist->FindOrBuildMaterial("G4_BONE_COMPACT_ICRU");
    Muscle = nist -> FindOrBuildMaterial("G4_MUSCLE_SKELETAL_ICRP");
    Adipose = nist -> FindOrBuildMaterial("G4_ADIPOSE_TISSUE_ICRP");
    Skin = nist -> FindOrBuildMaterial("G4_SKIN_ICRP");
    Muscle_Sucrose = nist -> FindOrBuildMaterial("G4_MUSCLE_WITH_SUCROSE");

    TissueMix = new G4Material("TissueMix", 1.036*g/cm3, 3); 
    TissueMix -> AddMaterial(Muscle, 79.36*perCent); 
    TissueMix -> AddMaterial(Adipose, 15.87*perCent); 
    TissueMix -> AddMaterial(Skin, 04.77*perCent);

    Light_Adipose = new G4Material("Light_Adipose", 0.6 * g / cm3, 1);
    Light_Adipose -> AddMaterial(Adipose, 100*perCent);

    OsBone  =  new G4Material("OsteoporoticBone", 0.80 *g/cm3, 8);
    OsBone -> AddMaterial(nist -> FindOrBuildMaterial("G4_H"),  06.4 * perCent);
    OsBone -> AddMaterial(nist -> FindOrBuildMaterial("G4_C"),  27.8 * perCent);
    OsBone -> AddMaterial(nist -> FindOrBuildMaterial("G4_N"),  02.7 * perCent);
    OsBone -> AddMaterial(nist -> FindOrBuildMaterial("G4_O"),  41.0 * perCent);
    OsBone -> AddMaterial(nist -> FindOrBuildMaterial("G4_Mg"), 00.2 * perCent);
    OsBone -> AddMaterial(nist -> FindOrBuildMaterial("G4_P"),  07.0 * perCent);
    OsBone -> AddMaterial(nist -> FindOrBuildMaterial("G4_S"),  00.2 * perCent);
    OsBone -> AddMaterial(nist -> FindOrBuildMaterial("G4_Ca"), 14.7 * perCent);

    // Aerogel and Material Proporties =================================================================================

    Aerogel = new G4Material("Aerogel", 10.000*g/cm3, 3);
    Aerogel -> AddMaterial(SiO2, 62.5 * perCent);
    Aerogel -> AddMaterial(H2O , 37.4 * perCent);
    Aerogel -> AddElement (C   , 00.1 * perCent);

    G4double PhotonEnergy[2] = {1.239841939 * eV/0.9, 1.239841939 * eV/0.2};
    G4double RindexAerogel[2] = {1.1, 1.1};
    G4double RindexWorld[2] = {1.0, 1.0};

    G4MaterialPropertiesTable * AerogelProperties = new G4MaterialPropertiesTable();
    AerogelProperties -> AddProperty("RINDEX", PhotonEnergy, RindexAerogel, 2);
    Aerogel -> SetMaterialPropertiesTable(AerogelProperties);
    
    G4MaterialPropertiesTable * worldMaterialProperties = new G4MaterialPropertiesTable();
    worldMaterialProperties -> AddProperty("RINDEX", PhotonEnergy, RindexWorld, 2);
    worldMaterial -> SetMaterialPropertiesTable(worldMaterialProperties);
}


/*G4double angleX = Model3DRotation.x() + 100 * deg;
G4double angleY = Model3DRotation.y() + 0 * deg;
G4double angleZ = Model3DRotation.z() + 5 * deg;
G4double angleX2 = Model3DRotation.x() - 100 * deg;
G4double angleY2 = Model3DRotation.y() + 0 * deg;
G4double angleZ2 = Model3DRotation.z() + 0 * deg;*/

//El vector de posicion del tumor con el substract tiene los ejes invertidos 
// tumor Y = substract Z
// tumor X = substract -X
// Colocar el tumor en el modelo de tórax
//new G4PVPlacement(Model3DRotation, Model3DRotation->inverse() *  tumorPosition, logicTumor, "Tumor", logicWorld, false, 0, true);