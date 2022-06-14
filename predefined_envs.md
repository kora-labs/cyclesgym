Here, we give an overview of the pre-registered environments and their naming convention

### N fertilization environments

The fertilization environments differ along the following dimensions: weather generation, location, duration.
All these environments are readily available and can be created with 
```
import gym
gym.make(id=env_id)
```        
where `env_id=f{'Corn{duration}{location}{weather}-v1'}`. Below, we explain the values that these variables can have and
their corresponding meaning.
- `duration` indicates time horizon of the experiment. It can take the values `Short` (1 year), `Middle` (2 years), 
or `Long` (5 years).
- `location` indicates the location where the experiment takes place,  which affects the historical weather data that 
is used. It can take the values `RockSprings` or `NewHolland`.
- `weather` indicates how the weather is generated, which can either be random (random shuffled years from historical 
- data) or fixed. It can take the values `RW` (random) or `FW` (fixed).

### Crop planning environments

The crop planning environments differ along the following dimensions: weather generation, location. The duration is 
fixed to 19 years as these experiments only make sense over long time horizons. All these environments are readily 
available and can be created with 
```
import gym
gym.make(id=env_id)
```        
where `env_id=f{'CropPlanning{location}{weather}-v1'}`. Below, we explain the values that these variables can have and
their corresponding meaning.
- `location` indicates the location where the experiment takes place,  which affects the historical weather data that 
is used. It can take the values `RockSprings` or `NewHolland`.
- `weather` indicates how the weather is generated, which can either be random (random shuffled years from historical 
data) or fixed. It can take the values `RW` (random) or `FW` (fixed).
