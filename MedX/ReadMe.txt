MedX is a Geant4 project oriented to the accurate simulation of radiographies. 

It has 5 modes of operation, mapping to the number of arguments in command line:

1. Launching the graphical interace.                                         Command Line: ./MedX 
2. Running the radiograph simulation in background console.                  Command Line: ./MedX example_macro.mac
3. Running the simulation that calculates the attenuation coefficient        Command Line: ./MedX example_macro.mac .
4. Running the x-ray tube simulation in background console.                  Command Line: ./MedX example_macro.mac . .
5. Running radiograph or CT (with reduced data saved) in background console. Command Line: ./MedX example_macro.mac . . .