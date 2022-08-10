## Interface
Cycles simulations are fully specified via a set of configuration files (see 
[Cycles documentation](https://psumodeling.github.io/Cycles/)). 
The most imporant files are the following:

 1. Control: It specifies the generic parameters of the simulation, e.g., the 
duration, the outputs to write, the weather, soil, crop, and management files 
to use.
 2. Weather: It specifies the weather in daily steps. It contains information 
 regarding temperature, humidity, radiation, wind, precipitation, wind, and 
 geographical location of the weather station.
 3. Soil: It contains several parameters characterizing each layer of the soil 
 at the start of the simulation.
 4. Crop: It contains the physiological parameters of the crops that will be 
 simulated. 
 5. Operation: It contains all the management practices that will be adopted 
 during the simulation, including tillage, irrigation, planting, and 
 fertilization. Each one of these macro operations is specified by several 
 parameters.
 
Similarly, Cycles returns the results of its simulation via a set of files 
including those describing the soil, water, and crop status daily.

To interact with Cycles, cyclesgym uses a set of managers that are implemented 
in `cyclesgym/managers`. Each type of file has a dedicated manager. Each 
manager can parse the corresponding file type and load it into an appropriate 
data structure (dictionary or pandas.Dataframe). The managers of input 
files can also write to file. This way, it is possible to create new weather 
conditions, soils, crops, and management files from Python.

