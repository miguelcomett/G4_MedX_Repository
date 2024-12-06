#include "3.2_Geometry3D.hh"

G4TessellatedSolid * G4STL::Read(const G4String & path, const G4String & name)
{
    FILE * fid = fopen(path.c_str(), "rb");
    if (fid == NULL) return NULL;

    G4TessellatedSolid * geometry = new G4TessellatedSolid(name.empty() ? path : name);
    if (geometry == NULL) return NULL;

    char buffer[80];
    if (fread(buffer, sizeof * buffer, 80, fid) != 80)
        goto error;

    uint32_t n;
    if (fread(&n, sizeof n, 1, fid) != 1)
        goto error;

    if (fVerbosity) 
    {
        G4cout << "G4STL - Loading file " << path << " which contains " << n << " facets" << G4endl;
    }

    for (uint32_t i = 0; i < n; i++) 
    {
        if (fread(buffer, sizeof * buffer, 50, fid) != 50)
            goto error;

        float * v0 = ((float *)buffer) + 3;
        float * v1 = v0 + 3;
        float * v2 = v1 + 3;

        G4TriangularFacet * facet = new G4TriangularFacet(
            G4ThreeVector(v0[0] * fUnit, v0[1] * fUnit, v0[2] * fUnit),
            G4ThreeVector(v1[0] * fUnit, v1[1] * fUnit, v1[2] * fUnit),
            G4ThreeVector(v2[0] * fUnit, v2[1] * fUnit, v2[2] * fUnit),
            ABSOLUTE);

        if (!geometry -> AddFacet( (G4VFacet *) facet) )
            goto error;
    }
    fclose(fid);

    geometry -> SetSolidClosed(true);

    return geometry;

error:
    fclose(fid);
    delete geometry;
    return NULL;
}