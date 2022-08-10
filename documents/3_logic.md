## Cyclesgym logic
Originally, Cycles does not allow the user to make interactive management 
decisions based on intermediate observations. Instead, it requires all 
management practices to be pre-specified, which makes it impossible to use it
in a reinforcement learning (RL) loop. Cyclesgym is meant to address this issue.
To this end, it creates all the necessary input files for the desired Cycles 
simulation, parses the output after the desired time interval has elapsed 
(usually one week but this can be changed), passes the relevant variables to 
the decision-maker (the RL agent, usually), updates the operation file if 
necessary, and repeats this procedure. 

The shortest possible simulation duration in Cycles is one year. Therefore, to 
implement the behavior described above between $t$ and $\delta t$, cyclesgym 
needs to:
1. Read the output at time t and pass it to the decision-maker.
2. Receive the new decision (which may include "do nothing") and update the 
operation file, if necessary.
3. If the operation file was updated, re-run the simulation and read the values 
in the  new output files at time $t+\delta t$. Otherwise, directly read the 
values in the old output files at time $t+\delta t$.  


**Speed considerations**: Due step 3 above, the speed of cyclesgym depends on (among other things):
 1. How often the decision-maker updates the operation files (each time triggers a 
new simulation). This means it is not easy to time cycles as the duration of 
one episode depends on the decisions being made. 
 2. The type of outputs that are required from Cycles. This is because some of
 the output files can be quite large and writing them to memory may be slow.   

 Point 1 has two important consequences for the training of RL agents. First, 
 as training progresses, policies become more stable, which induces faster 
 episodes. Second, making the time resolution finer can make the rollout of one 
 episode slower as there are more occasions to update the operation file 
 (notice that this matters early in the training when policies keep changing).

#### CyclesEnv and specific environments
The logic described above is common to all the environments, no matter the 
observation space, action space, rewards, and constraints. It is implemented in 
generic class `CyclesEnv` that is contained in `cyclesgym/envs/commmon.py`. 
This class contains the functionalities to create a unique temporary directory, 
writing, copying, or symlinking all the necessary input files there, 
initialize all the managers that are necessary to interact with Cycles, and run
Cycles simulations from Python. 

The states, actions, rewards, and constraints are defined in each individual 
environment. This is done by implementing the functions `_init_observer`,
`_init_implementer`, `_init_rewarder`, `_init_constrainer`. Cyclesgym comes 
with a set of predefined environments which already implement observers, 
implementers, rewarders, and constrainers for specific scenarios (see 
[here](3.1_predefined_envs.md) for a detailed description of the environments 
available). 

New environments can be created by creating new state spaces, action spaces, 
rewards, and/or constraints or by combining existing ones (see [here](3.3_custom_spaces_and_rewards.md)). 
Moreover, it is possible to modify the initial state distribution  using a custom soil generator (see [here](3.2_custom_weather_and_soil.md)). 
Finally,  the dynamics of the environment can be modified by using a custom weather generator
or different crops. You can find an in depth discussion on how to do this 
[here](3.2_custom_weather_and_soil.md).    