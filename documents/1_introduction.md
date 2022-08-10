## Introduction
The scientific community has developed several crop growth models (CGMs) and each one has 
its strengths and weaknesses. Therefore, which one to use depends on the 
research question under investigation and there is not a single best one [1].

Cycles is a mechanistic multi-year and multi-species agroecosystem model, 
which evolved from C-Farm [2] and shares biophysical modules with CropSyst 
[3]. It simulates the water and energy balance, the coupled cycling of carbon 
and nitrogen, and plant growth at daily time steps. Its ability to simulate a 
wide range of perturbations of biogeochemical processes caused by management 
practices for multiple crops and its focus on long-term simulations make it a 
suitable CGM to study the application of RL to sustainable agriculture, where 
these aspects are crucial.

In summary, our choice for Cycles is dictated by:
 1. Its accurate modeling of Nitrogen dynamics.
 2. Its focus on long-term simulations.
 3. Its ability to simulate multiple crops, making adept at modelling complex 
 agricultural systems and the resulting interactions including crop rotations 
 and polycultures.

#### Alternative RL environments for agricultural management
To the best of our knowledge, two other OpenAI gym environments based on crop 
growth models currently exist:
 1. [cropgym](https://github.com/BigDataWUR/crop-gym) based on LINTUL3.
 2. [gym-DSSAT](https://rgautron.gitlabpages.inria.fr/gym-dssat-docs/) based on DSSAT for maize.
 
Cropgym can simulate multiple crops but it is limited to single-year 
experiments. gym-DSSAT can simulate multiple-year scenarios but is limited to 
maize. Ultimately, the choice of environment depends on the research that 
is being pursued and on aspects that include modelling accuracy, speed, ease of 
customization, and more. Therefore, which environment is application 
dependent and there is no single best option.


 

## References
 [1] Di Paola, A., Valentini, R., and Santini, M., "An overview of available 
 crop growth and yield models for studies and assessments in agriculture." 
 Journal of the Science of Food and Agriculture 96.3 (2016): 709-714. 
 
 [2] Kemanian, A. R., Stöckle., C. O., "C-Farm: A simple model to evaluate the 
 carbon balance of soil profiles." European Journal of Agronomy 32.1 (2010): 22-29.
 
 [3] Stöckle, C. O., Donatelli, M., Nelson., R., "CropSyst, a cropping systems 
 simulation model." European journal of agronomy 18.3-4 (2003): 289-307.