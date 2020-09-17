# PHAS0077HaiyanLi

Background
---
Research on catalysts is always blossoming due to the increasing demand of improving efficiency and saving costs in productivity. The computational method Kinetic Monte Carlo exhibits excellent advantage in simulating and investigating chemical behaviours on catalytic surfaces. Thus, we employ the KMC to a few well-known chemical systems and non-reactive surface processes to explore its superiority.

Usage
---
The python files here can be divided into three categories. 

One is applicaitons of off-lattice KMC in a few well-known chemical systems under the guidance of Gillespie's paper. For each chemical system, 
we create a sperate python document, which contains the definition of a class and a main function. Once you run the python document, the main 
function will be executed automatically and outputs graphs that display simulation trajectories. All data used in these python docuements are in line with 
that in Gillespie's paper.

The second category is applications of on-lattice KMC. Depending on if rate constants of all types of events are the same, we design algorithms differently
and exploe how the coverage changes as simulation time moves on. 

The last category is applications of on-lattice KMC in non-reactive surface processes of CO. To compare the simulaiton results from KMC to 
the results calculated from the Langmuir isotherm, we create two classes. One is the Langmuir adsorption model, and the other is an on-lattice 
KMC model. In CoverageRelationshipPlot.py, the output is a graph which shows the relationship between the coverage of CO and its partial pressure under 
different temperatures.Therefore, we can select reasonable temperature and pressure as inputs of KMC simulation.By invoking KMCVSLangmuir.py, 
the average coverage from KMC simulation and the coverage from the Langmuir isotherm calculation will come out automatically. By invoking LangmuirIsothermCalculation.py, the coverage from the Langmuir isotherm calculation will come out.

Contributor 
---
@KrystalLiLi
