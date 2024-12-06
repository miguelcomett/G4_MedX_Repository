#ifndef STLGEOMETRYREADER_HH
#define STLGEOMETRYREADER_HH

#include "G4VSolid.hh"
#include <string>

class STLGeometryReader 
{
    public:
    
        STLGeometryReader() = default;
        ~STLGeometryReader() = default;
        G4VSolid * CreateSolidFromSTL(const std::string & filename);
};
#endif