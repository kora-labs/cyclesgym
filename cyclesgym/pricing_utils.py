# Conversion rate for corn from bushel to metric ton from
# https://grains.org/markets-tools-data/tools/converting-grain-units/
CORN_BUSHEL_PER_TONNE = 39.3680
SOYBEAN_BUSHEL_PER_TONNE = 36.7437

# Avg anhydrous ammonia cost in 2020 from
# https://farmdocdaily.illinois.edu/2021/08/2021-fertilizer-price-increases-in-perspective-with-implications-for-2022-costs.html
# Computed as 496 * 0.001 ($/ton * ton/kg)
N_price_dollars_per_kg = {y: 0.001 for y in range(1980, 2020)}

# Avg US price of corn for 2020 from
# https://quickstats.nass.usda.gov/results/BA8CCB81-A2BB-3C5C-BD23-DBAC365C7832
corn_price_dollars_per_bushel = {y: 4.53 for y in range(1980, 2020)}
corn_price_dollars_per_tonne = {y: CORN_BUSHEL_PER_TONNE * corn_price_dollars_per_bushel[y]
                                for y in corn_price_dollars_per_bushel.keys()}

# US price of corn silage (forage) in 1970 (not available more recently)
# https://quickstats.nass.usda.gov/results/6C3AADDF-25D2-31E7-9E0A-F050F345F91D
# https://beef.unl.edu/beefwatch/2020/corn-crop-worth-more-silage-or-grain#:~:text=Corn%20Silage%20Packed%20in%20the%20Silo&text=The%202020%20Nebraska%20Farm%20Custom,price%20per%20ton%20is%20%2434.95.
corn_silage_price_dollars_per_tonne = {y: 10 for y in range(1980, 2020)}

# Avg US price of soybean for 2020 from
# https://quickstats.nass.usda.gov/results/1A09097A-EFA4-3C47-B1D4-E7ACDFAA2575
soy_beans_price_dollars_per_bushel = {y: 9.89 for y in range(1980, 2020)}
soy_beans_price_dollars_per_tonne = {y: SOYBEAN_BUSHEL_PER_TONNE * soy_beans_price_dollars_per_bushel[y]
                                     for y in soy_beans_price_dollars_per_bushel.keys()}

#crop prices in $ per tonne
crop_prices = {'CornRM.90': corn_price_dollars_per_tonne,
               'SoybeanMG.5': soy_beans_price_dollars_per_tonne,
               'SoybeanMG.3': soy_beans_price_dollars_per_tonne,
               'CornSilageRM.90': corn_silage_price_dollars_per_tonne
               }

crop_type = {'CornRM.90': 'GRAIN YIELD',
             'CornSilageRM.90': 'FORAGE YIELD',
             'SoybeanMG.5': 'GRAIN YIELD',
             'SoybeanMG.3': 'GRAIN YIELD'
             }