# -*- coding: utf-8 -*-
"""
May 2019
@author: Sebastiaan Deetman; deetman@cml.leidenuniv.nl
contributions from: Sylvia Marinova
"""

#%% GENERAL SETTING & STATEMENTS
import pandas as pd
import numpy as np
import os
import ctypes     
import matplotlib.pyplot as plt
import math

# set current directory
os.chdir("C:\\Users\\...")   # SET YOUR PATH HERE

# Set general constants
regions = 26        #26 IMAGE regions
building_types = 4  #4 building types: detached, semi-detached, appartments & high-rise 
area = 2            #2 areas: rural & urban
materials = 7       #6 materials: Steel, Cement, Concrete, Wood, Copper, Aluminium, Glass
inflation = 1.2423  # gdp/cap inflation correction between 2005 (IMAGE data) & 2016 (commercial calibration) according to https://www.bls.gov/data/inflation_calculator.htm

# Set Flags for sensitivity analysis
flag_alpha = 0      # switch for the sensitivity analysis on alpha, if 1 the maximum alpha is 10% above the maximum found in the data
flag_ExpDec = 0     # switch to choose between Gompertz and Exponential Decay function for commercial floorspace demand (0 = Gompertz, 1 = Expdec)
flag_Normal = 1     # switch to choose between Weibull and Normal lifetime distributions (0 = Weibull, 1 = Normal)
flag_Mean   = 0     # switch to choose between material intensity settings (0 = regular regional, 1 = mean, 2 = high, 3 = low, 4 = median)

#%%Load files & arrange tables ----------------------------------------------------

if flag_Mean == 0:
    file_addition = ''
elif flag_Mean == 1:
    file_addition = '_mean'
elif flag_Mean ==2:
    file_addition = '_high'
elif flag_Mean ==3:
    file_addition = '_low'
else:
    file_addition = '_median'

# load material Databe csv-files
avg_m2_cap = pd.read_csv('files_DB\Average_m2_per_cap.csv')           # Avg_m2_cap; unit: m2/capita; meaning: average square meters per person (by region & rural/urban) 
building_materials = pd.read_csv('files_DB\Building_materials' + file_addition + '.csv')   # Building_materials; unit: kg/m2; meaning: the average material use per square meter (by building type, by region & by area)
housing_type = pd.read_csv('files_DB\Housing_type.csv')               # Housing_type; unit: %; meaning: the share of the NUMBER OF PEOPLE living in a particular building type (by region & by area) 
materials_commercial = pd.read_csv('files_DB\materials_commercial' + file_addition + '.csv', index_col = [0]) # 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type) 

# load IMAGE csv-files
floorspace = pd.read_csv('files_IMAGE/res_Floorspace.csv')                                  # Floorspace; unit: m2/capita; meaning: the average m2 per capita (over time, by region & area)
floorspace = floorspace[floorspace.Region != regions + 1]                                # Remove empty region 27
pop = pd.read_csv('files_IMAGE/pop.csv', index_col = [0])                                   # Pop; unit: million of people; meaning: global population (over time, by region)             
rurpop = pd.read_csv('files_IMAGE/rurpop.csv', index_col = [0])                             # rurpop; unit: %; meaning: the share of people living in rural areas (over time, by region)
sva_pc_2005 = pd.read_csv('files_IMAGE/sva_pc.csv', index_col = [0])
sva_pc = sva_pc_2005 * inflation                                                            # we use the inflation corrected SVA to adjust for the fact that IMAGE provides gdp/cap in 2005 US$

# Load fitted regression parameters
if flag_alpha == 0:
    gompertz = pd.read_csv('files_commercial/Gompertz_parameters.csv', index_col = [0])
else:
    gompertz = pd.read_csv('files_commercial/Gompertz_parameters_alpha.csv', index_col = [0])

# Ensure full time series  for pop & rurpop (interpolation, some years are missing)
rurpop2 = rurpop.reindex(list(range(1970,2051,1))).interpolate()
pop2 = pop.reindex(list(range(1970,2051,1))).interpolate()

# Remove 1st year, to ensure same Table size as floorspace data (from 1971)
pop2 = pop2.iloc[1:]
rurpop2 = rurpop2.iloc[1:]

#pre-calculate urban population
urbpop = 1 - rurpop2                                                           # urban population is 1 - the fraction of people living in rural areas (rurpop)
        
# Restructure the tables to regions as columns; for floorspace
floorspace_rur = floorspace.pivot(index="t", columns="Region", values="Rural")
floorspace_urb = floorspace.pivot(index="t", columns="Region", values="Urban")

# Restructuring for square meters (m2/cap)
avg_m2_cap_urb = avg_m2_cap.loc[avg_m2_cap['Area'] == 'Urban'].drop('Area', 1).T  # Remove area column & Transpose
avg_m2_cap_urb.columns = list(map(int,avg_m2_cap_urb.iloc[0]))                      # name columns according to the row containing the region-labels
avg_m2_cap_urb2 = avg_m2_cap_urb.drop(['Region'])                                 # Remove idle row 

avg_m2_cap_rur = avg_m2_cap.loc[avg_m2_cap['Area'] == 'Rural'].drop('Area', 1).T  # Remove area column & Transpose
avg_m2_cap_rur.columns = list(map(int,avg_m2_cap_rur.iloc[0]))                      # name columns according to the row containing the region-labels
avg_m2_cap_rur2 = avg_m2_cap_rur.drop(['Region'])                                 # Remove idle row 

# Restructuring for the Housing types (% of population living in them)
housing_type_urb = housing_type.loc[housing_type['Area'] == 'Urban'].drop('Area', 1).T  # Remove area column & Transpose
housing_type_urb.columns = list(map(int,housing_type_urb.iloc[0]))                      # name columns according to the row containing the region-labels
housing_type_urb2 = housing_type_urb.drop(['Region'])                                 # Remove idle row 

housing_type_rur = housing_type.loc[housing_type['Area'] == 'Rural'].drop('Area', 1).T  # Remove area column & Transpose
housing_type_rur.columns = list(map(int,housing_type_rur.iloc[0]))                      # name columns according to the row containing the region-labels
housing_type_rur2 = housing_type_rur.drop(['Region'])                                 # Remove idle row 

#%% COMMERCIAL building space demand (stock) calculated from Gomperz curve (fitted, using separate regression model)

# Select gompertz curve paramaters for the total commercial m2 demand (stock)
alpha = gompertz['All']['a'] if flag_ExpDec == 0 else 25.601
beta =  gompertz['All']['b'] if flag_ExpDec == 0 else 28.431
gamma = gompertz['All']['c'] if flag_ExpDec == 0 else 0.0415

# find the total commercial m2 stock (in Millions of m2)
commercial_m2_cap = pd.DataFrame(index=range(1971,2051), columns=range(1,27))
for year in range(1971,2051):
    for region in range(1,27):
        if flag_ExpDec == 0:
            commercial_m2_cap[region][year] = alpha * math.exp(-beta * math.exp((-gamma/1000) * sva_pc[str(region)][year]))
        else:
            commercial_m2_cap[region][year] = max(0.542, alpha - beta * math.exp((-gamma/1000) * sva_pc[str(region)][year]))

# Subdivide the total across Offices, Retail+, Govt+ & Hotels+
commercial_m2_cap_office = pd.DataFrame(index=range(1971,2051), columns=range(1,27))    # Offices
commercial_m2_cap_retail = pd.DataFrame(index=range(1971,2051), columns=range(1,27))    # Retail & Warehouses
commercial_m2_cap_hotels = pd.DataFrame(index=range(1971,2051), columns=range(1,27))    # Hotels & Restaurants
commercial_m2_cap_govern = pd.DataFrame(index=range(1971,2051), columns=range(1,27))    # Hospitals, Education, Government & Transportation

minimum_com_office = 25
minimum_com_retail = 25
minimum_com_hotels = 25
minimum_com_govern = 25

for year in range(1971,2051):
    for region in range(1,27):
        
        # get the square meter per capita floorspace for 4 commercial applications
        office = gompertz['Office']['a'] * math.exp(-gompertz['Office']['b'] * math.exp((-gompertz['Office']['c']/1000) * sva_pc[str(region)][year]))
        retail = gompertz['Retail+']['a'] * math.exp(-gompertz['Retail+']['b'] * math.exp((-gompertz['Retail+']['c']/1000) * sva_pc[str(region)][year]))
        hotels = gompertz['Hotels+']['a'] * math.exp(-gompertz['Hotels+']['b'] * math.exp((-gompertz['Hotels+']['c']/1000) * sva_pc[str(region)][year]))
        govern = gompertz['Govt+']['a'] * math.exp(-gompertz['Govt+']['b'] * math.exp((-gompertz['Govt+']['c']/1000) * sva_pc[str(region)][year]))

        #calculate minimum values for later use in historic tail(Region 20: China @ 134 $/cap SVA)
        minimum_com_office = office if office < minimum_com_office else minimum_com_office      
        minimum_com_retail = retail if retail < minimum_com_retail else minimum_com_retail
        minimum_com_hotels = hotels if hotels < minimum_com_hotels else minimum_com_hotels
        minimum_com_govern = govern if govern < minimum_com_govern else minimum_com_govern
        
        # Then use the ratio's to subdivide the total commercial floorspace into 4 categories      
        commercial_sum = office + retail + hotels + govern
        
        commercial_m2_cap_office[region][year] = commercial_m2_cap[region][year] * (office/commercial_sum)
        commercial_m2_cap_retail[region][year] = commercial_m2_cap[region][year] * (retail/commercial_sum)
        commercial_m2_cap_hotels[region][year] = commercial_m2_cap[region][year] * (hotels/commercial_sum)
        commercial_m2_cap_govern[region][year] = commercial_m2_cap[region][year] * (govern/commercial_sum)

#%% Add historic tail (1720-1970) + 100 yr initial --------------------------------------------

# load historic population development
hist_pop = pd.read_csv('files_initial_stock\hist_pop.csv', index_col = [0])  # initial population as a percentage of the 1970 population; unit: %; according to the Maddison Project Database (MPD) 2018 (Groningen University)

# Determine the historical average global trend in floorspace/cap  & the regional rural population share based on the last 10 years of IMAGE data
floorspace_urb_trend_by_region = [0 for j in range(0,26)]
floorspace_rur_trend_by_region = [0 for j in range(0,26)]
rurpop_trend_by_region = [0 for j in range(0,26)]
commercial_m2_cap_office_trend = [0 for j in range(0,26)]
commercial_m2_cap_retail_trend = [0 for j in range(0,26)]
commercial_m2_cap_hotels_trend = [0 for j in range(0,26)]
commercial_m2_cap_govern_trend = [0 for j in range(0,26)]

# For the RESIDENTIAL & COMMERCIAL floorspace: Derive the annual trend (in m2/cap) over the initial 10 years of IMAGE data
for region in range(1,27):
    floorspace_urb_trend_by_year = [0 for i in range(0,10)]
    floorspace_rur_trend_by_year = [0 for i in range(0,10)]
    commercial_m2_cap_office_trend_by_year = [0 for j in range(0,10)]    
    commercial_m2_cap_retail_trend_by_year = [0 for i in range(0,10)]   
    commercial_m2_cap_hotels_trend_by_year = [0 for j in range(0,10)]
    commercial_m2_cap_govern_trend_by_year = [0 for i in range(0,10)]
    
    # Get the growth by year (for the first 10 years)
    for year in range(1970,1980):
        floorspace_urb_trend_by_year[year-1970] = floorspace_urb[region][year+1]/floorspace_urb[region][year+2]
        floorspace_rur_trend_by_year[year-1970] = floorspace_rur[region][year+1]/floorspace_rur[region][year+2]
        commercial_m2_cap_office_trend_by_year[year-1970] = commercial_m2_cap_office[region][year+1]/commercial_m2_cap_office[region][year+2]
        commercial_m2_cap_retail_trend_by_year[year-1970] = commercial_m2_cap_retail[region][year+1]/commercial_m2_cap_retail[region][year+2] 
        commercial_m2_cap_hotels_trend_by_year[year-1970] = commercial_m2_cap_hotels[region][year+1]/commercial_m2_cap_hotels[region][year+2]
        commercial_m2_cap_govern_trend_by_year[year-1970] = commercial_m2_cap_govern[region][year+1]/commercial_m2_cap_govern[region][year+2]
        
    rurpop_trend_by_region[region-1] = ((1-(rurpop[str(region)][1980]/rurpop[str(region)][1970]))/10)*100
    floorspace_urb_trend_by_region[region-1] = sum(floorspace_urb_trend_by_year)/10
    floorspace_rur_trend_by_region[region-1] = sum(floorspace_rur_trend_by_year)/10
    commercial_m2_cap_office_trend[region-1] = sum(commercial_m2_cap_office_trend_by_year)/10
    commercial_m2_cap_retail_trend[region-1] = sum(commercial_m2_cap_retail_trend_by_year)/10
    commercial_m2_cap_hotels_trend[region-1] = sum(commercial_m2_cap_hotels_trend_by_year)/10
    commercial_m2_cap_govern_trend[region-1] = sum(commercial_m2_cap_govern_trend_by_year)/10

# Average global annual decline in floorspace/cap in %, rural: 1%; urban 1.2%;  commercial: 1.26-2.18% /yr   
floorspace_urb_trend_global = (1-(sum(floorspace_urb_trend_by_region)/26))*100              # in % decrease per annum
floorspace_rur_trend_global = (1-(sum(floorspace_rur_trend_by_region)/26))*100              # in % decrease per annum
commercial_m2_cap_office_trend_global = (1-(sum(commercial_m2_cap_office_trend)/26))*100    # in % decrease per annum
commercial_m2_cap_retail_trend_global = (1-(sum(commercial_m2_cap_retail_trend)/26))*100    # in % decrease per annum
commercial_m2_cap_hotels_trend_global = (1-(sum(commercial_m2_cap_hotels_trend)/26))*100    # in % decrease per annum
commercial_m2_cap_govern_trend_global = (1-(sum(commercial_m2_cap_govern_trend)/26))*100    # in % decrease per annum


# define historic floorspace (1820-1970) in m2/cap
floorspace_urb_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=floorspace_urb.columns)
floorspace_rur_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=floorspace_rur.columns)
rurpop_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=rurpop.columns)
pop_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=pop2.columns)
commercial_m2_cap_office_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_office.columns)
commercial_m2_cap_retail_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_retail.columns)
commercial_m2_cap_hotels_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_hotels.columns)
commercial_m2_cap_govern_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_govern.columns)

# Find minumum or maximum values in the original IMAGE data (Just for residential, commercial minimum values have been calculated above)
minimum_urb_fs = floorspace_urb.values.min()    # Region 20: China
minimum_rur_fs = floorspace_rur.values.min()    # Region 20: China
maximum_rurpop = rurpop.values.max()            # Region 9 : Eastern Africa

# Calculate the actual values used between 1820 & 1970, given the trends & the min/max values
for region in range(1,regions+1):
    for year in range(1820,1971):
        # MAX of 1) the MINimum value & 2) the calculated value
        floorspace_urb_1820_1970[region][year] = max(minimum_urb_fs, floorspace_urb[region][1971] * ((100-floorspace_urb_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        floorspace_rur_1820_1970[region][year] = max(minimum_rur_fs, floorspace_rur[region][1971] * ((100-floorspace_rur_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        commercial_m2_cap_office_1820_1970[region][year] = max(minimum_com_office, commercial_m2_cap_office[region][1971] * ((100-commercial_m2_cap_office_trend_global)/100)**(1971-year))  # single global value for average annual Decrease  
        commercial_m2_cap_retail_1820_1970[region][year] = max(minimum_com_retail, commercial_m2_cap_retail[region][1971] * ((100-commercial_m2_cap_retail_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        commercial_m2_cap_hotels_1820_1970[region][year] = max(minimum_com_hotels, commercial_m2_cap_hotels[region][1971] * ((100-commercial_m2_cap_hotels_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        commercial_m2_cap_govern_1820_1970[region][year] = max(minimum_com_govern, commercial_m2_cap_govern[region][1971] * ((100-commercial_m2_cap_govern_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        # MIN of 1) the MAXimum value & 2) the calculated value        
        rurpop_1820_1970[str(region)][year] = min(maximum_rurpop, rurpop[str(region)][1970] * ((100+rurpop_trend_by_region[region-1])/100)**(1970-year))  # average annual INcrease by region
        # just add the tail to the population (no min/max & trend is pre-calculated in hist_pop)        
        pop_1820_1970[str(region)][year] = hist_pop[str(region)][year] * pop[str(region)][1970]

urbpop_1820_1970 = 1 - rurpop_1820_1970

# To avoid full model setup in 1820 (all required stock gets built in yr 1) we assume another tail that linearly increases to the 1820 value over a 100 year time period, so 1720 = 0
floorspace_urb_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=floorspace_urb.columns)
floorspace_rur_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=floorspace_rur.columns)
rurpop_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=rurpop.columns)
urbpop_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=urbpop.columns)
pop_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=pop2.columns)
commercial_m2_cap_office_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=commercial_m2_cap_office.columns)
commercial_m2_cap_retail_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=commercial_m2_cap_retail.columns)
commercial_m2_cap_hotels_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=commercial_m2_cap_hotels.columns)
commercial_m2_cap_govern_1721_1820 = pd.DataFrame(index=range(1721,1820), columns=commercial_m2_cap_govern.columns)

for region in range(1,27):
    for time in range(1721,1820):
        #                                                        MAX(0,...) Because of floating point deviations, leading to negative stock in some cases
        floorspace_urb_1721_1820[int(region)][time]            = max(0.0, floorspace_urb_1820_1970[int(region)][1820] - (floorspace_urb_1820_1970[int(region)][1820]/100)*(1820-time))
        floorspace_rur_1721_1820[int(region)][time]            = max(0.0, floorspace_rur_1820_1970[int(region)][1820] - (floorspace_rur_1820_1970[int(region)][1820]/100)*(1820-time))
        rurpop_1721_1820[str(region)][time]                    = max(0.0, rurpop_1820_1970[str(region)][1820] - (rurpop_1820_1970[str(region)][1820]/100)*(1820-time))
        urbpop_1721_1820[str(region)][time]                    = max(0.0, urbpop_1820_1970[str(region)][1820] - (urbpop_1820_1970[str(region)][1820]/100)*(1820-time))
        pop_1721_1820[str(region)][time]                       = max(0.0, pop_1820_1970[str(region)][1820] - (pop_1820_1970[str(region)][1820]/100)*(1820-time))
        commercial_m2_cap_office_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_office_1820_1970[region][1820] - (commercial_m2_cap_office_1820_1970[region][1820]/100)*(1820-time))
        commercial_m2_cap_retail_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_retail_1820_1970[region][1820] - (commercial_m2_cap_retail_1820_1970[region][1820]/100)*(1820-time))
        commercial_m2_cap_hotels_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_hotels_1820_1970[region][1820] - (commercial_m2_cap_hotels_1820_1970[region][1820]/100)*(1820-time))
        commercial_m2_cap_govern_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_govern_1820_1970[region][1820] - (commercial_m2_cap_govern_1820_1970[region][1820]/100)*(1820-time))

# combine historic with IMAGE data here
rurpop_tail                     = rurpop_1820_1970.append(rurpop2, ignore_index=False)
urbpop_tail                     = urbpop_1820_1970.append(urbpop, ignore_index=False)
pop_tail                        = pop_1820_1970.append(pop2, ignore_index=False)
floorspace_urb_tail             = floorspace_urb_1820_1970.append(floorspace_urb, ignore_index=False)
floorspace_rur_tail             = floorspace_rur_1820_1970.append(floorspace_rur, ignore_index=False)
commercial_m2_cap_office_tail   = commercial_m2_cap_office_1820_1970.append(commercial_m2_cap_office, ignore_index=False)
commercial_m2_cap_retail_tail   = commercial_m2_cap_retail_1820_1970.append(commercial_m2_cap_retail, ignore_index=False)
commercial_m2_cap_hotels_tail   = commercial_m2_cap_hotels_1820_1970.append(commercial_m2_cap_hotels, ignore_index=False)
commercial_m2_cap_govern_tail   = commercial_m2_cap_govern_1820_1970.append(commercial_m2_cap_govern, ignore_index=False)

rurpop_tail                     = rurpop_1721_1820.append(rurpop_1820_1970.append(rurpop2, ignore_index=False), ignore_index=False)
urbpop_tail                     = urbpop_1721_1820.append(urbpop_1820_1970.append(urbpop, ignore_index=False), ignore_index=False)
pop_tail                        = pop_1721_1820.append(pop_1820_1970.append(pop2, ignore_index=False), ignore_index=False)
floorspace_urb_tail             = floorspace_urb_1721_1820.append(floorspace_urb_1820_1970.append(floorspace_urb, ignore_index=False), ignore_index=False)
floorspace_rur_tail             = floorspace_rur_1721_1820.append(floorspace_rur_1820_1970.append(floorspace_rur, ignore_index=False), ignore_index=False)
commercial_m2_cap_office_tail   = commercial_m2_cap_office_1721_1820.append(commercial_m2_cap_office_1820_1970.append(commercial_m2_cap_office, ignore_index=False), ignore_index=False)
commercial_m2_cap_retail_tail   = commercial_m2_cap_retail_1721_1820.append(commercial_m2_cap_retail_1820_1970.append(commercial_m2_cap_retail, ignore_index=False), ignore_index=False)
commercial_m2_cap_hotels_tail   = commercial_m2_cap_hotels_1721_1820.append(commercial_m2_cap_hotels_1820_1970.append(commercial_m2_cap_hotels, ignore_index=False), ignore_index=False)
commercial_m2_cap_govern_tail   = commercial_m2_cap_govern_1721_1820.append(commercial_m2_cap_govern_1820_1970.append(commercial_m2_cap_govern, ignore_index=False), ignore_index=False)

#%% SQUARE METER Calculations -----------------------------------------------------------

# adjust the share for urban/rural only (shares in csv are as percantage of the total(Rur + Urb), we needed to adjust the urban shares to add up to 1, same for rural)
housing_type_rur3 = housing_type_rur2/housing_type_rur2.sum()
housing_type_urb3 = housing_type_urb2/housing_type_urb2.sum()

# calculte the total rural/urban population (pop2 = millions of people, rurpop2 = % of people living in rural areas)
people_rur = pd.DataFrame(rurpop_tail.values*pop_tail.values, columns=pop_tail.columns, index=pop_tail.index)
people_urb = pd.DataFrame(urbpop_tail.values*pop_tail.values, columns=pop_tail.columns, index=pop_tail.index)

# calculate the total number of people (urban/rural) BY HOUSING TYPE (the sum of det,sem,app & hig equals the total population e.g. people_rur)
people_det_rur = pd.DataFrame(housing_type_rur3.iloc[0].values*people_rur.values, columns=people_rur.columns, index=people_rur.index)
people_sem_rur = pd.DataFrame(housing_type_rur3.iloc[1].values*people_rur.values, columns=people_rur.columns, index=people_rur.index)
people_app_rur = pd.DataFrame(housing_type_rur3.iloc[2].values*people_rur.values, columns=people_rur.columns, index=people_rur.index)
people_hig_rur = pd.DataFrame(housing_type_rur3.iloc[3].values*people_rur.values, columns=people_rur.columns, index=people_rur.index)

people_det_urb = pd.DataFrame(housing_type_urb3.iloc[0].values*people_urb.values, columns=people_urb.columns, index=people_urb.index)
people_sem_urb = pd.DataFrame(housing_type_urb3.iloc[1].values*people_urb.values, columns=people_urb.columns, index=people_urb.index)
people_app_urb = pd.DataFrame(housing_type_urb3.iloc[2].values*people_urb.values, columns=people_urb.columns, index=people_urb.index)
people_hig_urb = pd.DataFrame(housing_type_urb3.iloc[3].values*people_urb.values, columns=people_urb.columns, index=people_urb.index)

# calculate the total m2 (urban/rural) BY HOUSING TYPE (= nr. of people * OWN avg m2, so not based on IMAGE)
m2_unadjusted_det_rur = pd.DataFrame(avg_m2_cap_rur2.iloc[0].values * people_det_rur.values, columns=people_det_rur.columns, index=people_det_rur.index)
m2_unadjusted_sem_rur = pd.DataFrame(avg_m2_cap_rur2.iloc[1].values * people_sem_rur.values, columns=people_sem_rur.columns, index=people_sem_rur.index)
m2_unadjusted_app_rur = pd.DataFrame(avg_m2_cap_rur2.iloc[2].values * people_app_rur.values, columns=people_app_rur.columns, index=people_app_rur.index)
m2_unadjusted_hig_rur = pd.DataFrame(avg_m2_cap_rur2.iloc[3].values * people_hig_rur.values, columns=people_hig_rur.columns, index=people_hig_rur.index)

m2_unadjusted_det_urb = pd.DataFrame(avg_m2_cap_urb2.iloc[0].values * people_det_urb.values, columns=people_det_urb.columns, index=people_det_urb.index)
m2_unadjusted_sem_urb = pd.DataFrame(avg_m2_cap_urb2.iloc[1].values * people_sem_urb.values, columns=people_sem_urb.columns, index=people_sem_urb.index)
m2_unadjusted_app_urb = pd.DataFrame(avg_m2_cap_urb2.iloc[2].values * people_app_urb.values, columns=people_app_urb.columns, index=people_app_urb.index)
m2_unadjusted_hig_urb = pd.DataFrame(avg_m2_cap_urb2.iloc[3].values * people_hig_urb.values, columns=people_hig_urb.columns, index=people_hig_urb.index)

# Define empty dataframes for m2 adjustments
total_m2_adj_rur = pd.DataFrame(index=m2_unadjusted_det_rur.index, columns=m2_unadjusted_det_rur.columns)
total_m2_adj_urb = pd.DataFrame(index=m2_unadjusted_det_urb.index, columns=m2_unadjusted_det_urb.columns)

# Sum all square meters in Rural area
for j in range(1721,2051,1):
    for i in range(1,27,1):
        total_m2_adj_rur.loc[j,str(i)] = m2_unadjusted_det_rur.loc[j,str(i)] + m2_unadjusted_sem_rur.loc[j,str(i)] + m2_unadjusted_app_rur.loc[j,str(i)] + m2_unadjusted_hig_rur.loc[j,str(i)]

# Sum all square meters in Urban area
for j in range(1721,2051,1):
    for i in range(1,27,1):
        total_m2_adj_urb.loc[j,str(i)] = m2_unadjusted_det_urb.loc[j,str(i)] + m2_unadjusted_sem_urb.loc[j,str(i)] + m2_unadjusted_app_urb.loc[j,str(i)] + m2_unadjusted_hig_urb.loc[j,str(i)]

# average square meter per person implied by our OWN data
avg_m2_cap_adj_rur = pd.DataFrame(total_m2_adj_rur.values / people_rur.values, columns=people_rur.columns, index=people_rur.index) 
avg_m2_cap_adj_urb = pd.DataFrame(total_m2_adj_urb.values / people_urb.values, columns=people_urb.columns, index=people_urb.index)

# factor to correct square meters per capita so that we respect the IMAGE data in terms of total m2, but we use our own distinction between Building types
m2_cap_adj_fact_rur = pd.DataFrame(floorspace_rur_tail.values / avg_m2_cap_adj_rur.values, columns=floorspace_rur_tail.columns, index=floorspace_rur_tail.index)
m2_cap_adj_fact_urb = pd.DataFrame(floorspace_urb_tail.values / avg_m2_cap_adj_urb.values, columns=floorspace_urb_tail.columns, index=floorspace_urb_tail.index)

# All m2 by region (in millions), Building_type & year (using the correction factor, to comply with IMAGE avg m2/cap)
m2_det_rur = pd.DataFrame(m2_unadjusted_det_rur.values * m2_cap_adj_fact_rur.values, columns=m2_cap_adj_fact_rur.columns, index=m2_cap_adj_fact_rur.index)
m2_sem_rur = pd.DataFrame(m2_unadjusted_sem_rur.values * m2_cap_adj_fact_rur.values, columns=m2_cap_adj_fact_rur.columns, index=m2_cap_adj_fact_rur.index)
m2_app_rur = pd.DataFrame(m2_unadjusted_app_rur.values * m2_cap_adj_fact_rur.values, columns=m2_cap_adj_fact_rur.columns, index=m2_cap_adj_fact_rur.index)
m2_hig_rur = pd.DataFrame(m2_unadjusted_hig_rur.values * m2_cap_adj_fact_rur.values, columns=m2_cap_adj_fact_rur.columns, index=m2_cap_adj_fact_rur.index)

m2_det_urb = pd.DataFrame(m2_unadjusted_det_urb.values * m2_cap_adj_fact_urb.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
m2_sem_urb = pd.DataFrame(m2_unadjusted_sem_urb.values * m2_cap_adj_fact_urb.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
m2_app_urb = pd.DataFrame(m2_unadjusted_app_urb.values * m2_cap_adj_fact_urb.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
m2_hig_urb = pd.DataFrame(m2_unadjusted_hig_urb.values * m2_cap_adj_fact_urb.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)

# Add a checksum to see if calculations based on adjusted OWN avg m2 (by building type) now match the total m2 according to IMAGE. 
m2_sum_rur_OWN = m2_det_rur + m2_sem_rur + m2_app_rur + m2_hig_rur
m2_sum_rur_IMAGE = pd.DataFrame(floorspace_rur_tail.values*people_rur.values, columns=m2_sum_rur_OWN.columns, index=m2_sum_rur_OWN.index)
m2_checksum = m2_sum_rur_OWN - m2_sum_rur_IMAGE
if m2_checksum.sum().sum() > 0.0000001 or m2_checksum.sum().sum() < -0.0000001:
    ctypes.windll.user32.MessageBoxW(0, "IMAGE & OWN m2 sums do not match", "Warning", 1)

# total RESIDENTIAL square meters by region
m2 = m2_det_rur + m2_sem_rur + m2_app_rur + m2_hig_rur + m2_det_urb + m2_sem_urb + m2_app_urb + m2_hig_urb

# Total m2 for COMMERCIAL Buildings
commercial_m2_office = pd.DataFrame(commercial_m2_cap_office_tail.values * pop_tail.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
commercial_m2_retail = pd.DataFrame(commercial_m2_cap_retail_tail.values * pop_tail.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
commercial_m2_hotels = pd.DataFrame(commercial_m2_cap_hotels_tail.values * pop_tail.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)
commercial_m2_govern = pd.DataFrame(commercial_m2_cap_govern_tail.values * pop_tail.values, columns=m2_cap_adj_fact_urb.columns, index=m2_cap_adj_fact_urb.index)

#%% MATERIAL CALCULATIONS

# restructuring for the materials (kg/m2)
material_steel =    building_materials.pivot(index="Building_type", columns="Region", values="Steel")
material_cement =   building_materials.pivot(index="Building_type", columns="Region", values="Cement")
material_concrete = building_materials.pivot(index="Building_type", columns="Region", values="Concrete")
material_wood =      building_materials.pivot(index="Building_type", columns="Region", values="Wood")
material_copper =   building_materials.pivot(index="Building_type", columns="Region", values="Copper")
material_aluminium = building_materials.pivot(index="Building_type", columns="Region", values="Aluminium")
material_glass =    building_materials.pivot(index="Building_type", columns="Region", values="Glass")

# RURAL material stock (Millions of kgs = *1000 tons)
kg_det_rur_steel    = m2_det_rur * material_steel.loc[1]
kg_det_rur_cement   = m2_det_rur * material_cement.loc[1]
kg_det_rur_concrete = m2_det_rur * material_concrete.loc[1]
kg_det_rur_wood     = m2_det_rur * material_wood.loc[1]
kg_det_rur_copper   = m2_det_rur * material_copper.loc[1]
kg_det_rur_aluminium = m2_det_rur * material_aluminium.loc[1]
kg_det_rur_glass   = m2_det_rur * material_glass.loc[1]

kg_sem_rur_steel    = m2_sem_rur * material_steel.loc[2]
kg_sem_rur_cement   = m2_sem_rur * material_cement.loc[2]
kg_sem_rur_concrete = m2_sem_rur * material_concrete.loc[2]
kg_sem_rur_wood     = m2_sem_rur * material_wood.loc[2]
kg_sem_rur_copper   = m2_sem_rur * material_copper.loc[2]
kg_sem_rur_aluminium = m2_sem_rur * material_aluminium.loc[2]       # Adjusted in V2
kg_sem_rur_glass    = m2_sem_rur * material_glass.loc[2]            # Adjusted in V2

kg_app_rur_steel    = m2_app_rur * material_steel.loc[3]
kg_app_rur_cement   = m2_app_rur * material_cement.loc[3]
kg_app_rur_concrete = m2_app_rur * material_concrete.loc[3]
kg_app_rur_wood     = m2_app_rur * material_wood.loc[3]
kg_app_rur_copper   = m2_app_rur * material_copper.loc[3]
kg_app_rur_aluminium = m2_app_rur * material_aluminium.loc[3]       # Adjusted in V2
kg_app_rur_glass    = m2_app_rur * material_glass.loc[3]            # Adjusted in V2

kg_hig_rur_steel    = m2_hig_rur * material_steel.loc[4]
kg_hig_rur_cement   = m2_hig_rur * material_cement.loc[4]
kg_hig_rur_concrete = m2_hig_rur * material_concrete.loc[4]
kg_hig_rur_wood     = m2_hig_rur * material_wood.loc[4]
kg_hig_rur_copper   = m2_hig_rur * material_copper.loc[4]
kg_hig_rur_aluminium = m2_hig_rur * material_aluminium.loc[4]       # Adjusted in V2
kg_hig_rur_glass    = m2_hig_rur * material_glass.loc[4]            # Adjusted in V2

# URBAN material stock (millions of kgs)
kg_det_urb_steel    = m2_det_urb * material_steel.loc[1]
kg_det_urb_cement   = m2_det_urb * material_cement.loc[1]
kg_det_urb_concrete = m2_det_urb * material_concrete.loc[1]
kg_det_urb_wood     = m2_det_urb * material_wood.loc[1]
kg_det_urb_copper   = m2_det_urb * material_copper.loc[1]
kg_det_urb_aluminium  = m2_det_urb * material_aluminium.loc[1]
kg_det_urb_glass   = m2_det_urb * material_glass.loc[1]

kg_sem_urb_steel    = m2_sem_urb * material_steel.loc[2]
kg_sem_urb_cement   = m2_sem_urb * material_cement.loc[2]
kg_sem_urb_concrete = m2_sem_urb * material_concrete.loc[2]
kg_sem_urb_wood     = m2_sem_urb * material_wood.loc[2]
kg_sem_urb_copper   = m2_sem_urb * material_copper.loc[2]
kg_sem_urb_aluminium  = m2_sem_urb * material_aluminium.loc[2]      # Adjusted in V2
kg_sem_urb_glass   = m2_sem_urb * material_glass.loc[2]             # Adjusted in V2

kg_app_urb_steel    = m2_app_urb * material_steel.loc[3]
kg_app_urb_cement   = m2_app_urb * material_cement.loc[3]
kg_app_urb_concrete = m2_app_urb * material_concrete.loc[3]
kg_app_urb_wood     = m2_app_urb * material_wood.loc[3]
kg_app_urb_copper   = m2_app_urb * material_copper.loc[3]
kg_app_urb_aluminium  = m2_app_urb * material_aluminium.loc[3]      # Adjusted in V2
kg_app_urb_glass   = m2_app_urb * material_glass.loc[3]             # Adjusted in V2

kg_hig_urb_steel    = m2_hig_urb * material_steel.loc[4]
kg_hig_urb_cement   = m2_hig_urb * material_cement.loc[4]
kg_hig_urb_concrete = m2_hig_urb * material_concrete.loc[4]
kg_hig_urb_wood     = m2_hig_urb * material_wood.loc[4]
kg_hig_urb_copper   = m2_hig_urb * material_copper.loc[4]
kg_hig_urb_aluminium  = m2_hig_urb * material_aluminium.loc[4]      # Adjusted in V2
kg_hig_urb_glass   = m2_hig_urb * material_glass.loc[4]             # Adjusted in V2

# Commercial Building materials (in Million kg)
kg_office_steel     = commercial_m2_office * materials_commercial['Offices']['Steel']
kg_office_cement    = commercial_m2_office * materials_commercial['Offices']['Cement']
kg_office_concrete  = commercial_m2_office * materials_commercial['Offices']['Concrete']
kg_office_wood      = commercial_m2_office * materials_commercial['Offices']['Wood']
kg_office_copper    = commercial_m2_office * materials_commercial['Offices']['Copper']
kg_office_aluminium = commercial_m2_office * materials_commercial['Offices']['Aluminium']
kg_office_glass     = commercial_m2_office * materials_commercial['Offices']['Glass']

kg_retail_steel     = commercial_m2_retail * materials_commercial['Retail+']['Steel']
kg_retail_cement    = commercial_m2_retail * materials_commercial['Retail+']['Cement']
kg_retail_concrete  = commercial_m2_retail * materials_commercial['Retail+']['Concrete']
kg_retail_wood      = commercial_m2_retail * materials_commercial['Retail+']['Wood']
kg_retail_copper    = commercial_m2_retail * materials_commercial['Retail+']['Copper']
kg_retail_aluminium = commercial_m2_retail * materials_commercial['Retail+']['Aluminium']
kg_retail_glass     = commercial_m2_retail * materials_commercial['Retail+']['Glass']

kg_hotels_steel     = commercial_m2_hotels * materials_commercial['Hotels+']['Steel']
kg_hotels_cement    = commercial_m2_hotels * materials_commercial['Hotels+']['Cement']
kg_hotels_concrete  = commercial_m2_hotels * materials_commercial['Hotels+']['Concrete']
kg_hotels_wood      = commercial_m2_hotels * materials_commercial['Hotels+']['Wood']
kg_hotels_copper    = commercial_m2_hotels * materials_commercial['Hotels+']['Copper']
kg_hotels_aluminium = commercial_m2_hotels * materials_commercial['Hotels+']['Aluminium']
kg_hotels_glass     = commercial_m2_hotels * materials_commercial['Hotels+']['Glass']

kg_govern_steel     = commercial_m2_govern * materials_commercial['Govt+']['Steel']
kg_govern_cement    = commercial_m2_govern * materials_commercial['Govt+']['Cement']
kg_govern_concrete  = commercial_m2_govern * materials_commercial['Govt+']['Concrete']
kg_govern_wood      = commercial_m2_govern * materials_commercial['Govt+']['Wood']
kg_govern_copper    = commercial_m2_govern * materials_commercial['Govt+']['Copper']
kg_govern_aluminium = commercial_m2_govern * materials_commercial['Govt+']['Aluminium']
kg_govern_glass     = commercial_m2_govern * materials_commercial['Govt+']['Glass']

# Summing commercial material stock (Million kg)
kg_steel_comm       = kg_office_steel + kg_retail_steel + kg_hotels_steel + kg_govern_steel
kg_cement_comm      = kg_office_cement + kg_retail_cement + kg_hotels_cement + kg_govern_cement
kg_concrete_comm    = kg_office_concrete + kg_retail_concrete + kg_hotels_concrete + kg_govern_concrete
kg_wood_comm        = kg_office_wood + kg_retail_wood + kg_hotels_wood + kg_govern_wood
kg_copper_comm      = kg_office_copper + kg_retail_copper + kg_hotels_copper + kg_govern_copper
kg_aluminium_comm   = kg_office_aluminium + kg_retail_aluminium + kg_hotels_aluminium + kg_govern_aluminium
kg_glass_comm       = kg_office_glass + kg_retail_glass + kg_hotels_glass + kg_govern_glass

# Summing across RESIDENTIAL building types (millions of kg, in stock)
kg_steel_urb = kg_hig_urb_steel + kg_app_urb_steel + kg_sem_urb_steel + kg_det_urb_steel 
kg_steel_rur = kg_hig_rur_steel + kg_app_rur_steel + kg_sem_rur_steel + kg_det_rur_steel 

kg_cement_urb = kg_hig_urb_cement + kg_app_urb_cement + kg_sem_urb_cement + kg_det_urb_cement 
kg_cement_rur = kg_hig_rur_cement + kg_app_rur_cement + kg_sem_rur_cement + kg_det_rur_cement

kg_concrete_urb = kg_hig_urb_concrete + kg_app_urb_concrete + kg_sem_urb_concrete + kg_det_urb_concrete 
kg_concrete_rur = kg_hig_rur_concrete + kg_app_rur_concrete + kg_sem_rur_concrete + kg_det_rur_concrete

kg_wood_urb = kg_hig_urb_wood + kg_app_urb_wood + kg_sem_urb_wood + kg_det_urb_wood 
kg_wood_rur = kg_hig_rur_wood + kg_app_rur_wood + kg_sem_rur_wood + kg_det_rur_wood

kg_copper_urb = kg_hig_urb_copper + kg_app_urb_copper + kg_sem_urb_copper + kg_det_urb_copper 
kg_copper_rur = kg_hig_rur_copper + kg_app_rur_copper + kg_sem_rur_copper + kg_det_rur_copper

kg_aluminium_urb = kg_hig_urb_aluminium + kg_app_urb_aluminium + kg_sem_urb_aluminium + kg_det_urb_aluminium 
kg_aluminium_rur = kg_hig_rur_aluminium + kg_app_rur_aluminium + kg_sem_rur_aluminium + kg_det_rur_aluminium

kg_glass_urb = kg_hig_urb_glass + kg_app_urb_glass + kg_sem_urb_glass + kg_det_urb_glass 
kg_glass_rur = kg_hig_rur_glass + kg_app_rur_glass + kg_sem_rur_glass + kg_det_rur_glass

# Sums for total building material use (in-stock, millions of kg)
kg_steel    = kg_steel_urb + kg_steel_rur + kg_steel_comm
kg_cement   = kg_cement_urb + kg_cement_rur + kg_cement_comm
kg_concrete = kg_concrete_urb + kg_concrete_rur + kg_concrete_comm
kg_wood     = kg_wood_urb + kg_wood_rur + kg_wood_comm
kg_copper   = kg_copper_urb + kg_copper_rur + kg_copper_comm
kg_aluminium = kg_aluminium_urb + kg_aluminium_rur + kg_aluminium_comm
kg_glass   = kg_glass_urb + kg_glass_rur + kg_glass_comm


#%% INFLOW & OUTFLOW

import sys 
sys.path.append('C:\\Users\\...') # SET YOUR PATH HERE
import dynamic_stock_model   
from dynamic_stock_model import DynamicStockModel as DSM

if flag_Normal == 0:
    lifetimes_DB = pd.read_csv('files_lifetimes\lifetimes.csv')  # Weibull parameter database (shape & scale parameters given by region, area & building-type)
else:
    lifetimes_DB = pd.read_csv('files_lifetimes\lifetimes_normal.csv')  # Normal distribution database (Mean & StDev parameters given by region, area & building-type, though only defined by region for now)

# actual inflow calculations
def inflow_outflow(shape, scale, stock, length):            # length is the number of years in the entire period

    out_o_reg = pd.DataFrame(index=range(1721,2051), columns=range(1,27))
    out_i_reg = pd.DataFrame(index=range(1721,2051), columns=range(1,27))
    out_s_reg = pd.DataFrame(index=range(1721,2051), columns=range(1,27))
    
    for region in range(0,26):
        shape_list = [shape[region] for i in range(0,length)]    
        scale_list = [scale[region] for i in range(0,length)] 
        
        if flag_Normal == 0:
            DSMforward = DSM(t = np.arange(0,length,1), s=np.array(stock[region+1]), lt = {'Type': 'Weibull', 'Shape': np.array(shape_list), 'Scale': np.array(scale_list)})
        else:
            DSMforward = DSM(t = np.arange(0,length,1), s=np.array(stock[region+1]), lt = {'Type': 'FoldNorm', 'Mean': np.array(shape_list), 'StdDev': np.array(scale_list)}) # shape & scale list are actually Mean & StDev here
        
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)
        
        # (for now) We're only interested in the total outflow, so we sum the outflow by cohort each year
        out_o_reg[region+1] = out_oc.sum(axis=1)
        out_s_reg[region+1] = out_sc.sum(axis=1)
        out_o_reg_corr = out_o_reg._get_numeric_data()        
        out_o_reg_corr[out_o_reg_corr < 0] = 0            # remove negative outflow, replace by 0
        out_i_reg[region+1] = out_i
    
    return out_o_reg_corr, out_i_reg


length = len(m2_hig_urb[1])  # = 330

# the code to select the right shape & scale parameter from the database (lifetime_DB) is rather bulky, so we prepare a set of scale & shape parameters, instead of doing so 'in-line' when calling the stock model 
shape_selection_m2_det_rur = np.array(lifetimes_DB['Shape'].loc[(lifetimes_DB['Area'] == 'Rural') & (lifetimes_DB['Type'] == 'Detached')])
scale_selection_m2_det_rur = np.array(lifetimes_DB['Scale'].loc[(lifetimes_DB['Area'] == 'Rural') & (lifetimes_DB['Type'] == 'Detached')])
shape_selection_m2_sem_rur = np.array(lifetimes_DB['Shape'].loc[(lifetimes_DB['Area'] == 'Rural') & (lifetimes_DB['Type'] == 'Semi-detached')])
scale_selection_m2_sem_rur = np.array(lifetimes_DB['Scale'].loc[(lifetimes_DB['Area'] == 'Rural') & (lifetimes_DB['Type'] == 'Semi-detached')])
shape_selection_m2_app_rur = np.array(lifetimes_DB['Shape'].loc[(lifetimes_DB['Area'] == 'Rural') & (lifetimes_DB['Type'] == 'Appartments')])
scale_selection_m2_app_rur = np.array(lifetimes_DB['Scale'].loc[(lifetimes_DB['Area'] == 'Rural') & (lifetimes_DB['Type'] == 'Appartments')])
shape_selection_m2_hig_rur = np.array(lifetimes_DB['Shape'].loc[(lifetimes_DB['Area'] == 'Rural') & (lifetimes_DB['Type'] == 'High-rise')])
scale_selection_m2_hig_rur = np.array(lifetimes_DB['Scale'].loc[(lifetimes_DB['Area'] == 'Rural') & (lifetimes_DB['Type'] == 'High-rise')])
shape_selection_m2_det_urb = np.array(lifetimes_DB['Shape'].loc[(lifetimes_DB['Area'] == 'Urban') & (lifetimes_DB['Type'] == 'Detached')])
scale_selection_m2_det_urb = np.array(lifetimes_DB['Scale'].loc[(lifetimes_DB['Area'] == 'Urban') & (lifetimes_DB['Type'] == 'Detached')])
shape_selection_m2_sem_urb = np.array(lifetimes_DB['Shape'].loc[(lifetimes_DB['Area'] == 'Urban') & (lifetimes_DB['Type'] == 'Semi-detached')])
scale_selection_m2_sem_urb = np.array(lifetimes_DB['Scale'].loc[(lifetimes_DB['Area'] == 'Urban') & (lifetimes_DB['Type'] == 'Semi-detached')])
shape_selection_m2_app_urb = np.array(lifetimes_DB['Shape'].loc[(lifetimes_DB['Area'] == 'Urban') & (lifetimes_DB['Type'] == 'Appartments')])
scale_selection_m2_app_urb = np.array(lifetimes_DB['Scale'].loc[(lifetimes_DB['Area'] == 'Urban') & (lifetimes_DB['Type'] == 'Appartments')])
shape_selection_m2_hig_urb = np.array(lifetimes_DB['Shape'].loc[(lifetimes_DB['Area'] == 'Urban') & (lifetimes_DB['Type'] == 'High-rise')])
scale_selection_m2_hig_urb = np.array(lifetimes_DB['Scale'].loc[(lifetimes_DB['Area'] == 'Urban') & (lifetimes_DB['Type'] == 'High-rise')])

# Hardcoded lifetime parameters for COMMERCIAL building lifetime (avg. lt = 45 yr)
if flag_Normal == 0:
    scale_comm = np.array([49.567] * 26) # Weibull scale
    shape_comm = np.array([1.443] * 26)  # Weibull shape
else: 
    scale_comm = np.array([14] * 26)	# StDev in case of Normal distribution
    shape_comm = np.array([45] * 26)    # Mean in case of Normal distribution

# call the actual stock model to derive inflow & outflow based on stock & lifetime
m2_det_rur_o, m2_det_rur_i = inflow_outflow(shape_selection_m2_det_rur, scale_selection_m2_det_rur, m2_det_rur, length)
m2_sem_rur_o, m2_sem_rur_i = inflow_outflow(shape_selection_m2_sem_rur, scale_selection_m2_sem_rur, m2_sem_rur, length)
m2_app_rur_o, m2_app_rur_i = inflow_outflow(shape_selection_m2_app_rur, scale_selection_m2_app_rur, m2_app_rur, length)
m2_hig_rur_o, m2_hig_rur_i = inflow_outflow(shape_selection_m2_hig_rur, scale_selection_m2_hig_rur, m2_hig_rur, length)

m2_det_urb_o, m2_det_urb_i = inflow_outflow(shape_selection_m2_det_urb, scale_selection_m2_det_urb, m2_det_urb, length)
m2_sem_urb_o, m2_sem_urb_i = inflow_outflow(shape_selection_m2_sem_urb, scale_selection_m2_sem_urb, m2_sem_urb, length)
m2_app_urb_o, m2_app_urb_i = inflow_outflow(shape_selection_m2_app_urb, scale_selection_m2_app_urb, m2_app_urb, length)
m2_hig_urb_o, m2_hig_urb_i = inflow_outflow(shape_selection_m2_hig_urb, scale_selection_m2_hig_urb, m2_hig_urb, length)

m2_office_o, m2_office_i   = inflow_outflow(shape_comm, scale_comm, commercial_m2_office, length)
m2_retail_o, m2_retail_i   = inflow_outflow(shape_comm, scale_comm, commercial_m2_retail, length)
m2_hotels_o, m2_hotels_i   = inflow_outflow(shape_comm, scale_comm, commercial_m2_hotels, length)
m2_govern_o, m2_govern_i   = inflow_outflow(shape_comm, scale_comm, commercial_m2_govern, length)

# total MILLIONS of square meters inflow & outflow
m2_res_o = m2_det_rur_o + m2_sem_rur_o + m2_app_rur_o + m2_hig_rur_o + m2_det_urb_o + m2_sem_urb_o + m2_app_urb_o + m2_hig_urb_o
m2_res_i = m2_det_rur_i + m2_sem_rur_i + m2_app_rur_i + m2_hig_rur_i + m2_det_urb_i + m2_sem_urb_i + m2_app_urb_i + m2_hig_urb_i
m2_comm_o = m2_office_o + m2_retail_o + m2_hotels_o + m2_govern_o
m2_comm_i = m2_office_i + m2_retail_i + m2_hotels_i + m2_govern_i

#%% Material inflow & outflow

# RURAL material inlow (Millions of kgs = *1000 tons)
kg_det_rur_steel_i    = m2_det_rur_i * material_steel.loc[1]
kg_det_rur_cement_i   = m2_det_rur_i * material_cement.loc[1]
kg_det_rur_concrete_i = m2_det_rur_i * material_concrete.loc[1]
kg_det_rur_wood_i     = m2_det_rur_i * material_wood.loc[1]
kg_det_rur_copper_i   = m2_det_rur_i * material_copper.loc[1]
kg_det_rur_aluminium_i = m2_det_rur_i * material_aluminium.loc[1]
kg_det_rur_glass_i    = m2_det_rur_i * material_glass.loc[1]

kg_sem_rur_steel_i    = m2_sem_rur_i * material_steel.loc[2]
kg_sem_rur_cement_i   = m2_sem_rur_i * material_cement.loc[2]
kg_sem_rur_concrete_i = m2_sem_rur_i * material_concrete.loc[2]
kg_sem_rur_wood_i     = m2_sem_rur_i * material_wood.loc[2]
kg_sem_rur_copper_i   = m2_sem_rur_i * material_copper.loc[2]
kg_sem_rur_aluminium_i = m2_sem_rur_i * material_aluminium.loc[2]       # Adjusted in V2
kg_sem_rur_glass_i    = m2_sem_rur_i * material_glass.loc[2]            # Adjusted in V2

kg_app_rur_steel_i    = m2_app_rur_i * material_steel.loc[3]
kg_app_rur_cement_i   = m2_app_rur_i * material_cement.loc[3]
kg_app_rur_concrete_i = m2_app_rur_i * material_concrete.loc[3]
kg_app_rur_wood_i     = m2_app_rur_i * material_wood.loc[3]
kg_app_rur_copper_i   = m2_app_rur_i * material_copper.loc[3]
kg_app_rur_aluminium_i = m2_app_rur_i * material_aluminium.loc[3]       # Adjusted in V2
kg_app_rur_glass_i    = m2_app_rur_i * material_glass.loc[3]            # Adjusted in V2

kg_hig_rur_steel_i    = m2_hig_rur_i * material_steel.loc[4]
kg_hig_rur_cement_i   = m2_hig_rur_i * material_cement.loc[4]
kg_hig_rur_concrete_i = m2_hig_rur_i * material_concrete.loc[4]
kg_hig_rur_wood_i     = m2_hig_rur_i * material_wood.loc[4]
kg_hig_rur_copper_i   = m2_hig_rur_i * material_copper.loc[4]
kg_hig_rur_aluminium_i = m2_hig_rur_i * material_aluminium.loc[4]       # Adjusted in V2
kg_hig_rur_glass_i    = m2_hig_rur_i * material_glass.loc[4]            # Adjusted in V2

# URBAN material inflow (millions of kgs)
kg_det_urb_steel_i    = m2_det_urb_i * material_steel.loc[1]
kg_det_urb_cement_i   = m2_det_urb_i * material_cement.loc[1]
kg_det_urb_concrete_i = m2_det_urb_i * material_concrete.loc[1]
kg_det_urb_wood_i     = m2_det_urb_i * material_wood.loc[1]
kg_det_urb_copper_i   = m2_det_urb_i * material_copper.loc[1]
kg_det_urb_aluminium_i  = m2_det_urb_i * material_aluminium.loc[1]
kg_det_urb_glass_i   = m2_det_urb_i * material_glass.loc[1]

kg_sem_urb_steel_i    = m2_sem_urb_i * material_steel.loc[2]
kg_sem_urb_cement_i   = m2_sem_urb_i * material_cement.loc[2]
kg_sem_urb_concrete_i = m2_sem_urb_i * material_concrete.loc[2]
kg_sem_urb_wood_i     = m2_sem_urb_i * material_wood.loc[2]
kg_sem_urb_copper_i   = m2_sem_urb_i * material_copper.loc[2]
kg_sem_urb_aluminium_i  = m2_sem_urb_i * material_aluminium.loc[2]      # Adjusted in V2
kg_sem_urb_glass_i    = m2_sem_urb_i * material_glass.loc[2]            # Adjusted in V2

kg_app_urb_steel_i    = m2_app_urb_i * material_steel.loc[3]
kg_app_urb_cement_i   = m2_app_urb_i * material_cement.loc[3]
kg_app_urb_concrete_i = m2_app_urb_i * material_concrete.loc[3]
kg_app_urb_wood_i     = m2_app_urb_i * material_wood.loc[3]
kg_app_urb_copper_i   = m2_app_urb_i * material_copper.loc[3]
kg_app_urb_aluminium_i  = m2_app_urb_i * material_aluminium.loc[3]      # Adjusted in V2
kg_app_urb_glass_i   = m2_app_urb_i * material_glass.loc[3]             # Adjusted in V2

kg_hig_urb_steel_i    = m2_hig_urb_i * material_steel.loc[4]
kg_hig_urb_cement_i   = m2_hig_urb_i * material_cement.loc[4]
kg_hig_urb_concrete_i = m2_hig_urb_i * material_concrete.loc[4]
kg_hig_urb_wood_i     = m2_hig_urb_i * material_wood.loc[4]
kg_hig_urb_copper_i   = m2_hig_urb_i * material_copper.loc[4]
kg_hig_urb_aluminium_i  = m2_hig_urb_i * material_aluminium.loc[4]      # Adjusted in V2
kg_hig_urb_glass_i   = m2_hig_urb_i * material_glass.loc[4]             # Adjusted in V2

# RURAL material OUTflow (Millions of kgs = *1000 tons)
kg_det_rur_steel_o    = m2_det_rur_o * material_steel.loc[1]
kg_det_rur_cement_o   = m2_det_rur_o * material_cement.loc[1]
kg_det_rur_concrete_o = m2_det_rur_o * material_concrete.loc[1]
kg_det_rur_wood_o     = m2_det_rur_o * material_wood.loc[1]
kg_det_rur_copper_o   = m2_det_rur_o * material_copper.loc[1]
kg_det_rur_aluminium_o = m2_det_rur_o * material_aluminium.loc[1]
kg_det_rur_glass_o    = m2_det_rur_o * material_glass.loc[1]

kg_sem_rur_steel_o    = m2_sem_rur_o * material_steel.loc[2]
kg_sem_rur_cement_o   = m2_sem_rur_o * material_cement.loc[2]
kg_sem_rur_concrete_o = m2_sem_rur_o * material_concrete.loc[2]
kg_sem_rur_wood_o     = m2_sem_rur_o * material_wood.loc[2]
kg_sem_rur_copper_o   = m2_sem_rur_o * material_copper.loc[2]
kg_sem_rur_aluminium_o = m2_sem_rur_o * material_aluminium.loc[2]       # Adjusted in V2
kg_sem_rur_glass_o    = m2_sem_rur_o * material_glass.loc[2]            # Adjusted in V2

kg_app_rur_steel_o    = m2_app_rur_o * material_steel.loc[3]
kg_app_rur_cement_o   = m2_app_rur_o * material_cement.loc[3]
kg_app_rur_concrete_o = m2_app_rur_o * material_concrete.loc[3]
kg_app_rur_wood_o     = m2_app_rur_o * material_wood.loc[3]
kg_app_rur_copper_o   = m2_app_rur_o * material_copper.loc[3]
kg_app_rur_aluminium_o = m2_app_rur_o * material_aluminium.loc[3]       # Adjusted in V2
kg_app_rur_glass_o    = m2_app_rur_o * material_glass.loc[3]            # Adjusted in V2

kg_hig_rur_steel_o    = m2_hig_rur_o * material_steel.loc[4]
kg_hig_rur_cement_o   = m2_hig_rur_o * material_cement.loc[4]
kg_hig_rur_concrete_o = m2_hig_rur_o * material_concrete.loc[4]
kg_hig_rur_wood_o     = m2_hig_rur_o * material_wood.loc[4]
kg_hig_rur_copper_o   = m2_hig_rur_o * material_copper.loc[4]
kg_hig_rur_aluminium_o = m2_hig_rur_o * material_aluminium.loc[4]       # Adjusted in V2
kg_hig_rur_glass_o    = m2_hig_rur_o * material_glass.loc[4]            # Adjusted in V2

# URBAN material OUTflow (millions of kgs)
kg_det_urb_steel_o    = m2_det_urb_o * material_steel.loc[1]
kg_det_urb_cement_o   = m2_det_urb_o * material_cement.loc[1]
kg_det_urb_concrete_o = m2_det_urb_o * material_concrete.loc[1]
kg_det_urb_wood_o     = m2_det_urb_o * material_wood.loc[1]
kg_det_urb_copper_o   = m2_det_urb_o * material_copper.loc[1]
kg_det_urb_aluminium_o  = m2_det_urb_o * material_aluminium.loc[1]
kg_det_urb_glass_o   = m2_det_urb_o * material_glass.loc[1]

kg_sem_urb_steel_o    = m2_sem_urb_o * material_steel.loc[2]
kg_sem_urb_cement_o   = m2_sem_urb_o * material_cement.loc[2]
kg_sem_urb_concrete_o = m2_sem_urb_o * material_concrete.loc[2]
kg_sem_urb_wood_o     = m2_sem_urb_o * material_wood.loc[2]
kg_sem_urb_copper_o   = m2_sem_urb_o * material_copper.loc[2]
kg_sem_urb_aluminium_o  = m2_sem_urb_o * material_aluminium.loc[2]      # Adjusted in V2
kg_sem_urb_glass_o    = m2_sem_urb_o * material_glass.loc[2]            # Adjusted in V2

kg_app_urb_steel_o    = m2_app_urb_o * material_steel.loc[3]
kg_app_urb_cement_o   = m2_app_urb_o * material_cement.loc[3]
kg_app_urb_concrete_o = m2_app_urb_o * material_concrete.loc[3]
kg_app_urb_wood_o     = m2_app_urb_o * material_wood.loc[3]
kg_app_urb_copper_o   = m2_app_urb_o * material_copper.loc[3]
kg_app_urb_aluminium_o  = m2_app_urb_o * material_aluminium.loc[3]      # Adjusted in V2
kg_app_urb_glass_o   = m2_app_urb_o * material_glass.loc[3]             # Adjusted in V2

kg_hig_urb_steel_o    = m2_hig_urb_o * material_steel.loc[4]
kg_hig_urb_cement_o   = m2_hig_urb_o * material_cement.loc[4]
kg_hig_urb_concrete_o = m2_hig_urb_o * material_concrete.loc[4]
kg_hig_urb_wood_o     = m2_hig_urb_o * material_wood.loc[4]
kg_hig_urb_copper_o   = m2_hig_urb_o * material_copper.loc[4]
kg_hig_urb_aluminium_o  = m2_hig_urb_o * material_aluminium.loc[4]      # Adjusted in V2
kg_hig_urb_glass_o   = m2_hig_urb_o * material_glass.loc[4]             # Adjusted in V2

# Commercial Building materials INFLOW (in Million kg)
kg_office_steel_i     = m2_office_i * materials_commercial['Offices']['Steel']
kg_office_cement_i    = m2_office_i * materials_commercial['Offices']['Cement']
kg_office_concrete_i  = m2_office_i * materials_commercial['Offices']['Concrete']
kg_office_wood_i      = m2_office_i * materials_commercial['Offices']['Wood']
kg_office_copper_i    = m2_office_i * materials_commercial['Offices']['Copper']
kg_office_aluminium_i = m2_office_i * materials_commercial['Offices']['Aluminium']
kg_office_glass_i     = m2_office_i * materials_commercial['Offices']['Glass']

kg_retail_steel_i     = m2_retail_i * materials_commercial['Retail+']['Steel']
kg_retail_cement_i    = m2_retail_i * materials_commercial['Retail+']['Cement']
kg_retail_concrete_i  = m2_retail_i * materials_commercial['Retail+']['Concrete']
kg_retail_wood_i      = m2_retail_i * materials_commercial['Retail+']['Wood']
kg_retail_copper_i    = m2_retail_i * materials_commercial['Retail+']['Copper']
kg_retail_aluminium_i = m2_retail_i * materials_commercial['Retail+']['Aluminium']
kg_retail_glass_i     = m2_retail_i * materials_commercial['Retail+']['Glass']

kg_hotels_steel_i     = m2_hotels_i * materials_commercial['Hotels+']['Steel']
kg_hotels_cement_i    = m2_hotels_i * materials_commercial['Hotels+']['Cement']
kg_hotels_concrete_i  = m2_hotels_i * materials_commercial['Hotels+']['Concrete']
kg_hotels_wood_i      = m2_hotels_i * materials_commercial['Hotels+']['Wood']
kg_hotels_copper_i    = m2_hotels_i * materials_commercial['Hotels+']['Copper']
kg_hotels_aluminium_i = m2_hotels_i * materials_commercial['Hotels+']['Aluminium']
kg_hotels_glass_i     = m2_hotels_i * materials_commercial['Hotels+']['Glass']

kg_govern_steel_i     = m2_govern_i * materials_commercial['Govt+']['Steel']
kg_govern_cement_i    = m2_govern_i * materials_commercial['Govt+']['Cement']
kg_govern_concrete_i  = m2_govern_i * materials_commercial['Govt+']['Concrete']
kg_govern_wood_i      = m2_govern_i * materials_commercial['Govt+']['Wood']
kg_govern_copper_i    = m2_govern_i * materials_commercial['Govt+']['Copper']
kg_govern_aluminium_i = m2_govern_i * materials_commercial['Govt+']['Aluminium']
kg_govern_glass_i     = m2_govern_i * materials_commercial['Govt+']['Glass']

# Commercial Building materials OUTFLOW (in Million kg)
kg_office_steel_o     = m2_office_o * materials_commercial['Offices']['Steel']
kg_office_cement_o    = m2_office_o * materials_commercial['Offices']['Cement']
kg_office_concrete_o  = m2_office_o * materials_commercial['Offices']['Concrete']
kg_office_wood_o      = m2_office_o * materials_commercial['Offices']['Wood']
kg_office_copper_o    = m2_office_o * materials_commercial['Offices']['Copper']
kg_office_aluminium_o = m2_office_o * materials_commercial['Offices']['Aluminium']
kg_office_glass_o     = m2_office_o * materials_commercial['Offices']['Glass']

kg_retail_steel_o     = m2_retail_o * materials_commercial['Retail+']['Steel']
kg_retail_cement_o    = m2_retail_o * materials_commercial['Retail+']['Cement']
kg_retail_concrete_o  = m2_retail_o * materials_commercial['Retail+']['Concrete']
kg_retail_wood_o      = m2_retail_o * materials_commercial['Retail+']['Wood']
kg_retail_copper_o    = m2_retail_o * materials_commercial['Retail+']['Copper']
kg_retail_aluminium_o = m2_retail_o * materials_commercial['Retail+']['Aluminium']
kg_retail_glass_o     = m2_retail_o * materials_commercial['Retail+']['Glass']

kg_hotels_steel_o     = m2_hotels_o * materials_commercial['Hotels+']['Steel']
kg_hotels_cement_o    = m2_hotels_o * materials_commercial['Hotels+']['Cement']
kg_hotels_concrete_o  = m2_hotels_o * materials_commercial['Hotels+']['Concrete']
kg_hotels_wood_o      = m2_hotels_o * materials_commercial['Hotels+']['Wood']
kg_hotels_copper_o    = m2_hotels_o * materials_commercial['Hotels+']['Copper']
kg_hotels_aluminium_o = m2_hotels_o * materials_commercial['Hotels+']['Aluminium']
kg_hotels_glass_o     = m2_hotels_o * materials_commercial['Hotels+']['Glass']

kg_govern_steel_o     = m2_govern_o * materials_commercial['Govt+']['Steel']
kg_govern_cement_o    = m2_govern_o * materials_commercial['Govt+']['Cement']
kg_govern_concrete_o  = m2_govern_o * materials_commercial['Govt+']['Concrete']
kg_govern_wood_o      = m2_govern_o * materials_commercial['Govt+']['Wood']
kg_govern_copper_o    = m2_govern_o * materials_commercial['Govt+']['Copper']
kg_govern_aluminium_o = m2_govern_o * materials_commercial['Govt+']['Aluminium']
kg_govern_glass_o     = m2_govern_o * materials_commercial['Govt+']['Glass']


#%% CSV output (material stock & m2 stock)

length = 3
tag = ['stock', 'inflow', 'outflow']
  
# first, transpose all variables & add columns to identify material, area & appartment type
# RURAL
kg_det_rur_steel_out  = [[]] * length
kg_det_rur_steel_out[0]  = kg_det_rur_steel.transpose()
kg_det_rur_steel_out[1]  = kg_det_rur_steel_i.transpose()
kg_det_rur_steel_out[2]  = kg_det_rur_steel_o.transpose()
for item in range(0,length):
    kg_det_rur_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_det_rur_steel_out[item].insert(0,'area', ['rural'] * 26)
    kg_det_rur_steel_out[item].insert(0,'type', ['detached'] * 26)
    kg_det_rur_steel_out[item].insert(0,'flow', [tag[item]] * 26)
    
kg_det_rur_cement_out      = [[]] * length  
kg_det_rur_cement_out[0]   = kg_det_rur_cement.transpose() 
kg_det_rur_cement_out[1]   = kg_det_rur_cement_i.transpose() 
kg_det_rur_cement_out[2]   = kg_det_rur_cement_o.transpose() 
for item in range(0,length):
        kg_det_rur_cement_out[item].insert(0,'material', ['cement'] * 26)
        kg_det_rur_cement_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_cement_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_cement_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_det_rur_concrete_out      = [[]] * length  
kg_det_rur_concrete_out[0]   = kg_det_rur_concrete.transpose() 
kg_det_rur_concrete_out[1]   = kg_det_rur_concrete_i.transpose() 
kg_det_rur_concrete_out[2]   = kg_det_rur_concrete_o.transpose() 
for item in range(0,length):
        kg_det_rur_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_det_rur_concrete_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_concrete_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_det_rur_wood_out      = [[]] * length  
kg_det_rur_wood_out[0]   = kg_det_rur_wood.transpose() 
kg_det_rur_wood_out[1]   = kg_det_rur_wood_i.transpose() 
kg_det_rur_wood_out[2]   = kg_det_rur_wood_o.transpose() 
for item in range(0,length):
        kg_det_rur_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_det_rur_wood_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_wood_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_wood_out[item].insert(0,'flow', [tag[item]] * 26)
   
kg_det_rur_copper_out      = [[]] * length  
kg_det_rur_copper_out[0]   = kg_det_rur_copper.transpose() 
kg_det_rur_copper_out[1]   = kg_det_rur_copper_i.transpose() 
kg_det_rur_copper_out[2]   = kg_det_rur_copper_o.transpose() 
for item in range(0,length):
        kg_det_rur_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_det_rur_copper_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_copper_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_det_rur_aluminium_out      = [[]] * length  
kg_det_rur_aluminium_out[0]   = kg_det_rur_aluminium.transpose() 
kg_det_rur_aluminium_out[1]   = kg_det_rur_aluminium_i.transpose() 
kg_det_rur_aluminium_out[2]   = kg_det_rur_aluminium_o.transpose() 
for item in range(0,length):
        kg_det_rur_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_det_rur_aluminium_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_aluminium_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_det_rur_glass_out      = [[]] * length  
kg_det_rur_glass_out[0]   = kg_det_rur_glass.transpose() 
kg_det_rur_glass_out[1]   = kg_det_rur_glass_i.transpose() 
kg_det_rur_glass_out[2]   = kg_det_rur_glass_o.transpose() 
for item in range(0,length):
        kg_det_rur_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_det_rur_glass_out[item].insert(0,'area', ['rural'] * 26)
        kg_det_rur_glass_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_rur_glass_out[item].insert(0,'flow', [tag[item]] * 26)

kg_sem_rur_steel_out      = [[]] * length  
kg_sem_rur_steel_out[0]   = kg_sem_rur_steel.transpose() 
kg_sem_rur_steel_out[1]   = kg_sem_rur_steel_i.transpose() 
kg_sem_rur_steel_out[2]   = kg_sem_rur_steel_o.transpose() 
for item in range(0,length):
        kg_sem_rur_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_sem_rur_steel_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_steel_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_cement_out      = [[]] * length  
kg_sem_rur_cement_out[0]   = kg_sem_rur_cement.transpose() 
kg_sem_rur_cement_out[1]   = kg_sem_rur_cement_i.transpose() 
kg_sem_rur_cement_out[2]   = kg_sem_rur_cement_o.transpose() 
for item in range(0,length):
        kg_sem_rur_cement_out[item].insert(0,'material', ['cement'] * 26)
        kg_sem_rur_cement_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_cement_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_cement_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_concrete_out      = [[]] * length  
kg_sem_rur_concrete_out[0]   = kg_sem_rur_concrete.transpose() 
kg_sem_rur_concrete_out[1]   = kg_sem_rur_concrete_i.transpose() 
kg_sem_rur_concrete_out[2]   = kg_sem_rur_concrete_o.transpose() 
for item in range(0,length):
        kg_sem_rur_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_sem_rur_concrete_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_concrete_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_wood_out      = [[]] * length  
kg_sem_rur_wood_out[0]   = kg_sem_rur_wood.transpose() 
kg_sem_rur_wood_out[1]   = kg_sem_rur_wood_i.transpose() 
kg_sem_rur_wood_out[2]   = kg_sem_rur_wood_o.transpose() 
for item in range(0,length):
        kg_sem_rur_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_sem_rur_wood_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_wood_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_copper_out      = [[]] * length  
kg_sem_rur_copper_out[0]   = kg_sem_rur_copper.transpose() 
kg_sem_rur_copper_out[1]   = kg_sem_rur_copper_i.transpose() 
kg_sem_rur_copper_out[2]   = kg_sem_rur_copper_o.transpose() 
for item in range(0,length):
        kg_sem_rur_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_sem_rur_copper_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_copper_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_aluminium_out      = [[]] * length  
kg_sem_rur_aluminium_out[0]   = kg_sem_rur_aluminium.transpose() 
kg_sem_rur_aluminium_out[1]   = kg_sem_rur_aluminium_i.transpose() 
kg_sem_rur_aluminium_out[2]   = kg_sem_rur_aluminium_o.transpose() 
for item in range(0,length):
        kg_sem_rur_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_sem_rur_aluminium_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_aluminium_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_rur_glass_out      = [[]] * length  
kg_sem_rur_glass_out[0]   = kg_sem_rur_glass.transpose() 
kg_sem_rur_glass_out[1]   = kg_sem_rur_glass_i.transpose() 
kg_sem_rur_glass_out[2]   = kg_sem_rur_glass_o.transpose() 
for item in range(0,length):
        kg_sem_rur_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_sem_rur_glass_out[item].insert(0,'area', ['rural'] * 26)
        kg_sem_rur_glass_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_rur_glass_out[item].insert(0,'flow', [tag[item]] * 26)
 
#
kg_app_rur_steel_out      = [[]] * length  
kg_app_rur_steel_out[0]   = kg_app_rur_steel.transpose() 
kg_app_rur_steel_out[1]   = kg_app_rur_steel_i.transpose() 
kg_app_rur_steel_out[2]   = kg_app_rur_steel_o.transpose() 
for item in range(0,length):
        kg_app_rur_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_app_rur_steel_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_steel_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_cement_out      = [[]] * length  
kg_app_rur_cement_out[0]   = kg_app_rur_cement.transpose() 
kg_app_rur_cement_out[1]   = kg_app_rur_cement_i.transpose() 
kg_app_rur_cement_out[2]   = kg_app_rur_cement_o.transpose() 
for item in range(0,length):
        kg_app_rur_cement_out[item].insert(0,'material', ['cement'] * 26)
        kg_app_rur_cement_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_cement_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_cement_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_concrete_out      = [[]] * length  
kg_app_rur_concrete_out[0]   = kg_app_rur_concrete.transpose() 
kg_app_rur_concrete_out[1]   = kg_app_rur_concrete_i.transpose() 
kg_app_rur_concrete_out[2]   = kg_app_rur_concrete_o.transpose() 
for item in range(0,length):
        kg_app_rur_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_app_rur_concrete_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_concrete_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_wood_out      = [[]] * length  
kg_app_rur_wood_out[0]   = kg_app_rur_wood.transpose() 
kg_app_rur_wood_out[1]   = kg_app_rur_wood_i.transpose() 
kg_app_rur_wood_out[2]   = kg_app_rur_wood_o.transpose() 
for item in range(0,length):
        kg_app_rur_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_app_rur_wood_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_wood_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_copper_out      = [[]] * length  
kg_app_rur_copper_out[0]   = kg_app_rur_copper.transpose() 
kg_app_rur_copper_out[1]   = kg_app_rur_copper_i.transpose() 
kg_app_rur_copper_out[2]   = kg_app_rur_copper_o.transpose() 
for item in range(0,length):
        kg_app_rur_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_app_rur_copper_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_copper_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_aluminium_out      = [[]] * length  
kg_app_rur_aluminium_out[0]   = kg_app_rur_aluminium.transpose() 
kg_app_rur_aluminium_out[1]   = kg_app_rur_aluminium_i.transpose() 
kg_app_rur_aluminium_out[2]   = kg_app_rur_aluminium_o.transpose() 
for item in range(0,length):
        kg_app_rur_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_app_rur_aluminium_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_aluminium_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_rur_glass_out      = [[]] * length  
kg_app_rur_glass_out[0]   = kg_app_rur_glass.transpose() 
kg_app_rur_glass_out[1]   = kg_app_rur_glass_i.transpose() 
kg_app_rur_glass_out[2]   = kg_app_rur_glass_o.transpose() 
for item in range(0,length):
        kg_app_rur_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_app_rur_glass_out[item].insert(0,'area', ['rural'] * 26)
        kg_app_rur_glass_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_rur_glass_out[item].insert(0,'flow', [tag[item]] * 26)
        
kg_hig_rur_steel_out      = [[]] * length  
kg_hig_rur_steel_out[0]   = kg_hig_rur_steel.transpose() 
kg_hig_rur_steel_out[1]   = kg_hig_rur_steel_i.transpose() 
kg_hig_rur_steel_out[2]   = kg_hig_rur_steel_o.transpose() 
for item in range(0,length):
        kg_hig_rur_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_hig_rur_steel_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_steel_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_cement_out      = [[]] * length  
kg_hig_rur_cement_out[0]   = kg_hig_rur_cement.transpose() 
kg_hig_rur_cement_out[1]   = kg_hig_rur_cement_i.transpose() 
kg_hig_rur_cement_out[2]   = kg_hig_rur_cement_o.transpose() 
for item in range(0,length):
        kg_hig_rur_cement_out[item].insert(0,'material', ['cement'] * 26)
        kg_hig_rur_cement_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_cement_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_cement_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_concrete_out      = [[]] * length  
kg_hig_rur_concrete_out[0]   = kg_hig_rur_concrete.transpose() 
kg_hig_rur_concrete_out[1]   = kg_hig_rur_concrete_i.transpose() 
kg_hig_rur_concrete_out[2]   = kg_hig_rur_concrete_o.transpose() 
for item in range(0,length):
        kg_hig_rur_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_hig_rur_concrete_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_concrete_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_wood_out      = [[]] * length  
kg_hig_rur_wood_out[0]   = kg_hig_rur_wood.transpose() 
kg_hig_rur_wood_out[1]   = kg_hig_rur_wood_i.transpose() 
kg_hig_rur_wood_out[2]   = kg_hig_rur_wood_o.transpose() 
for item in range(0,length):
        kg_hig_rur_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_hig_rur_wood_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_wood_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_copper_out      = [[]] * length  
kg_hig_rur_copper_out[0]   = kg_hig_rur_copper.transpose() 
kg_hig_rur_copper_out[1]   = kg_hig_rur_copper_i.transpose() 
kg_hig_rur_copper_out[2]   = kg_hig_rur_copper_o.transpose() 
for item in range(0,length):
        kg_hig_rur_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_hig_rur_copper_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_copper_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_aluminium_out      = [[]] * length  
kg_hig_rur_aluminium_out[0]   = kg_hig_rur_aluminium.transpose() 
kg_hig_rur_aluminium_out[1]   = kg_hig_rur_aluminium_i.transpose() 
kg_hig_rur_aluminium_out[2]   = kg_hig_rur_aluminium_o.transpose() 
for item in range(0,length):
        kg_hig_rur_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_hig_rur_aluminium_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_aluminium_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_rur_glass_out      = [[]] * length  
kg_hig_rur_glass_out[0]   = kg_hig_rur_glass.transpose() 
kg_hig_rur_glass_out[1]   = kg_hig_rur_glass_i.transpose() 
kg_hig_rur_glass_out[2]   = kg_hig_rur_glass_o.transpose() 
for item in range(0,length):
        kg_hig_rur_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_hig_rur_glass_out[item].insert(0,'area', ['rural'] * 26)
        kg_hig_rur_glass_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_rur_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# URBAN 
kg_det_urb_steel_out  = [[]] * length
kg_det_urb_steel_out[0]  = kg_det_urb_steel.transpose()
kg_det_urb_steel_out[1]  = kg_det_urb_steel_i.transpose()
kg_det_urb_steel_out[2]  = kg_det_urb_steel_o.transpose()
for item in range(0,length):
    kg_det_urb_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_det_urb_steel_out[item].insert(0,'area', ['urban'] * 26)
    kg_det_urb_steel_out[item].insert(0,'type', ['detached'] * 26)
    kg_det_urb_steel_out[item].insert(0,'flow', [tag[item]] * 26)
    
kg_det_urb_cement_out      = [[]] * length  
kg_det_urb_cement_out[0]   = kg_det_urb_cement.transpose() 
kg_det_urb_cement_out[1]   = kg_det_urb_cement_i.transpose() 
kg_det_urb_cement_out[2]   = kg_det_urb_cement_o.transpose() 
for item in range(0,length):
        kg_det_urb_cement_out[item].insert(0,'material', ['cement'] * 26)
        kg_det_urb_cement_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_cement_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_cement_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_det_urb_concrete_out      = [[]] * length  
kg_det_urb_concrete_out[0]   = kg_det_urb_concrete.transpose() 
kg_det_urb_concrete_out[1]   = kg_det_urb_concrete_i.transpose() 
kg_det_urb_concrete_out[2]   = kg_det_urb_concrete_o.transpose() 
for item in range(0,length):
        kg_det_urb_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_det_urb_concrete_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_concrete_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_det_urb_wood_out      = [[]] * length  
kg_det_urb_wood_out[0]   = kg_det_urb_wood.transpose() 
kg_det_urb_wood_out[1]   = kg_det_urb_wood_i.transpose() 
kg_det_urb_wood_out[2]   = kg_det_urb_wood_o.transpose() 
for item in range(0,length):
        kg_det_urb_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_det_urb_wood_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_wood_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_wood_out[item].insert(0,'flow', [tag[item]] * 26)
   
kg_det_urb_copper_out      = [[]] * length  
kg_det_urb_copper_out[0]   = kg_det_urb_copper.transpose() 
kg_det_urb_copper_out[1]   = kg_det_urb_copper_i.transpose() 
kg_det_urb_copper_out[2]   = kg_det_urb_copper_o.transpose() 
for item in range(0,length):
        kg_det_urb_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_det_urb_copper_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_copper_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_det_urb_aluminium_out      = [[]] * length  
kg_det_urb_aluminium_out[0]   = kg_det_urb_aluminium.transpose() 
kg_det_urb_aluminium_out[1]   = kg_det_urb_aluminium_i.transpose() 
kg_det_urb_aluminium_out[2]   = kg_det_urb_aluminium_o.transpose() 
for item in range(0,length):
        kg_det_urb_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_det_urb_aluminium_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_aluminium_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_det_urb_glass_out      = [[]] * length  
kg_det_urb_glass_out[0]   = kg_det_urb_glass.transpose() 
kg_det_urb_glass_out[1]   = kg_det_urb_glass_i.transpose() 
kg_det_urb_glass_out[2]   = kg_det_urb_glass_o.transpose() 
for item in range(0,length):
        kg_det_urb_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_det_urb_glass_out[item].insert(0,'area', ['urban'] * 26)
        kg_det_urb_glass_out[item].insert(0,'type', ['detached'] * 26)
        kg_det_urb_glass_out[item].insert(0,'flow', [tag[item]] * 26)

kg_sem_urb_steel_out      = [[]] * length  
kg_sem_urb_steel_out[0]   = kg_sem_urb_steel.transpose() 
kg_sem_urb_steel_out[1]   = kg_sem_urb_steel_i.transpose() 
kg_sem_urb_steel_out[2]   = kg_sem_urb_steel_o.transpose() 
for item in range(0,length):
        kg_sem_urb_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_sem_urb_steel_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_steel_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_cement_out      = [[]] * length  
kg_sem_urb_cement_out[0]   = kg_sem_urb_cement.transpose() 
kg_sem_urb_cement_out[1]   = kg_sem_urb_cement_i.transpose() 
kg_sem_urb_cement_out[2]   = kg_sem_urb_cement_o.transpose() 
for item in range(0,length):
        kg_sem_urb_cement_out[item].insert(0,'material', ['cement'] * 26)
        kg_sem_urb_cement_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_cement_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_cement_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_concrete_out      = [[]] * length  
kg_sem_urb_concrete_out[0]   = kg_sem_urb_concrete.transpose() 
kg_sem_urb_concrete_out[1]   = kg_sem_urb_concrete_i.transpose() 
kg_sem_urb_concrete_out[2]   = kg_sem_urb_concrete_o.transpose() 
for item in range(0,length):
        kg_sem_urb_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_sem_urb_concrete_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_concrete_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_wood_out      = [[]] * length  
kg_sem_urb_wood_out[0]   = kg_sem_urb_wood.transpose() 
kg_sem_urb_wood_out[1]   = kg_sem_urb_wood_i.transpose() 
kg_sem_urb_wood_out[2]   = kg_sem_urb_wood_o.transpose() 
for item in range(0,length):
        kg_sem_urb_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_sem_urb_wood_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_wood_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_copper_out      = [[]] * length  
kg_sem_urb_copper_out[0]   = kg_sem_urb_copper.transpose() 
kg_sem_urb_copper_out[1]   = kg_sem_urb_copper_i.transpose() 
kg_sem_urb_copper_out[2]   = kg_sem_urb_copper_o.transpose() 
for item in range(0,length):
        kg_sem_urb_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_sem_urb_copper_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_copper_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_aluminium_out      = [[]] * length  
kg_sem_urb_aluminium_out[0]   = kg_sem_urb_aluminium.transpose() 
kg_sem_urb_aluminium_out[1]   = kg_sem_urb_aluminium_i.transpose() 
kg_sem_urb_aluminium_out[2]   = kg_sem_urb_aluminium_o.transpose() 
for item in range(0,length):
        kg_sem_urb_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_sem_urb_aluminium_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_aluminium_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_sem_urb_glass_out      = [[]] * length  
kg_sem_urb_glass_out[0]   = kg_sem_urb_glass.transpose() 
kg_sem_urb_glass_out[1]   = kg_sem_urb_glass_i.transpose() 
kg_sem_urb_glass_out[2]   = kg_sem_urb_glass_o.transpose() 
for item in range(0,length):
        kg_sem_urb_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_sem_urb_glass_out[item].insert(0,'area', ['urban'] * 26)
        kg_sem_urb_glass_out[item].insert(0,'type', ['semi-detached'] * 26)
        kg_sem_urb_glass_out[item].insert(0,'flow', [tag[item]] * 26)
 
#
kg_app_urb_steel_out      = [[]] * length  
kg_app_urb_steel_out[0]   = kg_app_urb_steel.transpose() 
kg_app_urb_steel_out[1]   = kg_app_urb_steel_i.transpose() 
kg_app_urb_steel_out[2]   = kg_app_urb_steel_o.transpose() 
for item in range(0,length):
        kg_app_urb_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_app_urb_steel_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_steel_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_cement_out      = [[]] * length  
kg_app_urb_cement_out[0]   = kg_app_urb_cement.transpose() 
kg_app_urb_cement_out[1]   = kg_app_urb_cement_i.transpose() 
kg_app_urb_cement_out[2]   = kg_app_urb_cement_o.transpose() 
for item in range(0,length):
        kg_app_urb_cement_out[item].insert(0,'material', ['cement'] * 26)
        kg_app_urb_cement_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_cement_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_cement_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_concrete_out      = [[]] * length  
kg_app_urb_concrete_out[0]   = kg_app_urb_concrete.transpose() 
kg_app_urb_concrete_out[1]   = kg_app_urb_concrete_i.transpose() 
kg_app_urb_concrete_out[2]   = kg_app_urb_concrete_o.transpose() 
for item in range(0,length):
        kg_app_urb_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_app_urb_concrete_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_concrete_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_wood_out      = [[]] * length  
kg_app_urb_wood_out[0]   = kg_app_urb_wood.transpose() 
kg_app_urb_wood_out[1]   = kg_app_urb_wood_i.transpose() 
kg_app_urb_wood_out[2]   = kg_app_urb_wood_o.transpose() 
for item in range(0,length):
        kg_app_urb_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_app_urb_wood_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_wood_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_copper_out      = [[]] * length  
kg_app_urb_copper_out[0]   = kg_app_urb_copper.transpose() 
kg_app_urb_copper_out[1]   = kg_app_urb_copper_i.transpose() 
kg_app_urb_copper_out[2]   = kg_app_urb_copper_o.transpose() 
for item in range(0,length):
        kg_app_urb_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_app_urb_copper_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_copper_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_aluminium_out      = [[]] * length  
kg_app_urb_aluminium_out[0]   = kg_app_urb_aluminium.transpose() 
kg_app_urb_aluminium_out[1]   = kg_app_urb_aluminium_i.transpose() 
kg_app_urb_aluminium_out[2]   = kg_app_urb_aluminium_o.transpose() 
for item in range(0,length):
        kg_app_urb_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_app_urb_aluminium_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_aluminium_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_app_urb_glass_out      = [[]] * length  
kg_app_urb_glass_out[0]   = kg_app_urb_glass.transpose() 
kg_app_urb_glass_out[1]   = kg_app_urb_glass_i.transpose() 
kg_app_urb_glass_out[2]   = kg_app_urb_glass_o.transpose() 
for item in range(0,length):
        kg_app_urb_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_app_urb_glass_out[item].insert(0,'area', ['urban'] * 26)
        kg_app_urb_glass_out[item].insert(0,'type', ['appartments'] * 26)
        kg_app_urb_glass_out[item].insert(0,'flow', [tag[item]] * 26)
        
kg_hig_urb_steel_out      = [[]] * length  
kg_hig_urb_steel_out[0]   = kg_hig_urb_steel.transpose() 
kg_hig_urb_steel_out[1]   = kg_hig_urb_steel_i.transpose() 
kg_hig_urb_steel_out[2]   = kg_hig_urb_steel_o.transpose() 
for item in range(0,length):
        kg_hig_urb_steel_out[item].insert(0,'material', ['steel'] * 26)
        kg_hig_urb_steel_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_steel_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_steel_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_cement_out      = [[]] * length  
kg_hig_urb_cement_out[0]   = kg_hig_urb_cement.transpose() 
kg_hig_urb_cement_out[1]   = kg_hig_urb_cement_i.transpose() 
kg_hig_urb_cement_out[2]   = kg_hig_urb_cement_o.transpose() 
for item in range(0,length):
        kg_hig_urb_cement_out[item].insert(0,'material', ['cement'] * 26)
        kg_hig_urb_cement_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_cement_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_cement_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_concrete_out      = [[]] * length  
kg_hig_urb_concrete_out[0]   = kg_hig_urb_concrete.transpose() 
kg_hig_urb_concrete_out[1]   = kg_hig_urb_concrete_i.transpose() 
kg_hig_urb_concrete_out[2]   = kg_hig_urb_concrete_o.transpose() 
for item in range(0,length):
        kg_hig_urb_concrete_out[item].insert(0,'material', ['concrete'] * 26)
        kg_hig_urb_concrete_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_concrete_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_concrete_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_wood_out      = [[]] * length  
kg_hig_urb_wood_out[0]   = kg_hig_urb_wood.transpose() 
kg_hig_urb_wood_out[1]   = kg_hig_urb_wood_i.transpose() 
kg_hig_urb_wood_out[2]   = kg_hig_urb_wood_o.transpose() 
for item in range(0,length):
        kg_hig_urb_wood_out[item].insert(0,'material', ['wood'] * 26)
        kg_hig_urb_wood_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_wood_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_wood_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_copper_out      = [[]] * length  
kg_hig_urb_copper_out[0]   = kg_hig_urb_copper.transpose() 
kg_hig_urb_copper_out[1]   = kg_hig_urb_copper_i.transpose() 
kg_hig_urb_copper_out[2]   = kg_hig_urb_copper_o.transpose() 
for item in range(0,length):
        kg_hig_urb_copper_out[item].insert(0,'material', ['copper'] * 26)
        kg_hig_urb_copper_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_copper_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_copper_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_aluminium_out      = [[]] * length  
kg_hig_urb_aluminium_out[0]   = kg_hig_urb_aluminium.transpose() 
kg_hig_urb_aluminium_out[1]   = kg_hig_urb_aluminium_i.transpose() 
kg_hig_urb_aluminium_out[2]   = kg_hig_urb_aluminium_o.transpose() 
for item in range(0,length):
        kg_hig_urb_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
        kg_hig_urb_aluminium_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_aluminium_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)
 
kg_hig_urb_glass_out      = [[]] * length  
kg_hig_urb_glass_out[0]   = kg_hig_urb_glass.transpose() 
kg_hig_urb_glass_out[1]   = kg_hig_urb_glass_i.transpose() 
kg_hig_urb_glass_out[2]   = kg_hig_urb_glass_o.transpose() 
for item in range(0,length):
        kg_hig_urb_glass_out[item].insert(0,'material', ['glass'] * 26)
        kg_hig_urb_glass_out[item].insert(0,'area', ['urban'] * 26)
        kg_hig_urb_glass_out[item].insert(0,'type', ['high-rise'] * 26)
        kg_hig_urb_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# COMMERCIAL ------------------------------------------------------------------

# offices
kg_office_steel_out  = [[]] * length
kg_office_steel_out[0]  = kg_office_steel.transpose()
kg_office_steel_out[1]  = kg_office_steel_i.transpose()
kg_office_steel_out[2]  = kg_office_steel_o.transpose()
for item in range(0,length):
    kg_office_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_office_steel_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_steel_out[item].insert(0,'type', ['office'] * 26)
    kg_office_steel_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_cement_out  = [[]] * length
kg_office_cement_out[0]  = kg_office_cement.transpose()
kg_office_cement_out[1]  = kg_office_cement_i.transpose()
kg_office_cement_out[2]  = kg_office_cement_o.transpose()
for item in range(0,length):
    kg_office_cement_out[item].insert(0,'material', ['cement'] * 26)
    kg_office_cement_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_cement_out[item].insert(0,'type', ['office'] * 26)
    kg_office_cement_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_concrete_out  = [[]] * length
kg_office_concrete_out[0]  = kg_office_concrete.transpose()
kg_office_concrete_out[1]  = kg_office_concrete_i.transpose()
kg_office_concrete_out[2]  = kg_office_concrete_o.transpose()
for item in range(0,length):
    kg_office_concrete_out[item].insert(0,'material', ['concrete'] * 26)
    kg_office_concrete_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_concrete_out[item].insert(0,'type', ['office'] * 26)
    kg_office_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_wood_out  = [[]] * length
kg_office_wood_out[0]  = kg_office_wood.transpose()
kg_office_wood_out[1]  = kg_office_wood_i.transpose()
kg_office_wood_out[2]  = kg_office_wood_o.transpose()
for item in range(0,length):
    kg_office_wood_out[item].insert(0,'material', ['wood'] * 26)
    kg_office_wood_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_wood_out[item].insert(0,'type', ['office'] * 26)
    kg_office_wood_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_copper_out  = [[]] * length
kg_office_copper_out[0]  = kg_office_copper.transpose()
kg_office_copper_out[1]  = kg_office_copper_i.transpose()
kg_office_copper_out[2]  = kg_office_copper_o.transpose()
for item in range(0,length):
    kg_office_copper_out[item].insert(0,'material', ['copper'] * 26)
    kg_office_copper_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_copper_out[item].insert(0,'type', ['office'] * 26)
    kg_office_copper_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_aluminium_out  = [[]] * length
kg_office_aluminium_out[0]  = kg_office_aluminium.transpose()
kg_office_aluminium_out[1]  = kg_office_aluminium_i.transpose()
kg_office_aluminium_out[2]  = kg_office_aluminium_o.transpose()
for item in range(0,length):
    kg_office_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
    kg_office_aluminium_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_aluminium_out[item].insert(0,'type', ['office'] * 26)
    kg_office_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_office_glass_out  = [[]] * length
kg_office_glass_out[0]  = kg_office_glass.transpose()
kg_office_glass_out[1]  = kg_office_glass_i.transpose()
kg_office_glass_out[2]  = kg_office_glass_o.transpose()
for item in range(0,length):
    kg_office_glass_out[item].insert(0,'material', ['glass'] * 26)
    kg_office_glass_out[item].insert(0,'area', ['commercial'] * 26)
    kg_office_glass_out[item].insert(0,'type', ['office'] * 26)
    kg_office_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# shops & retail
kg_retail_steel_out  = [[]] * length
kg_retail_steel_out[0]  = kg_retail_steel.transpose()
kg_retail_steel_out[1]  = kg_retail_steel_i.transpose()
kg_retail_steel_out[2]  = kg_retail_steel_o.transpose()
for item in range(0,length):
    kg_retail_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_retail_steel_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_steel_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_steel_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_cement_out  = [[]] * length
kg_retail_cement_out[0]  = kg_retail_cement.transpose()
kg_retail_cement_out[1]  = kg_retail_cement_i.transpose()
kg_retail_cement_out[2]  = kg_retail_cement_o.transpose()
for item in range(0,length):
    kg_retail_cement_out[item].insert(0,'material', ['cement'] * 26)
    kg_retail_cement_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_cement_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_cement_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_concrete_out  = [[]] * length
kg_retail_concrete_out[0]  = kg_retail_concrete.transpose()
kg_retail_concrete_out[1]  = kg_retail_concrete_i.transpose()
kg_retail_concrete_out[2]  = kg_retail_concrete_o.transpose()
for item in range(0,length):
    kg_retail_concrete_out[item].insert(0,'material', ['concrete'] * 26)
    kg_retail_concrete_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_concrete_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_wood_out  = [[]] * length
kg_retail_wood_out[0]  = kg_retail_wood.transpose()
kg_retail_wood_out[1]  = kg_retail_wood_i.transpose()
kg_retail_wood_out[2]  = kg_retail_wood_o.transpose()
for item in range(0,length):
    kg_retail_wood_out[item].insert(0,'material', ['wood'] * 26)
    kg_retail_wood_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_wood_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_wood_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_copper_out  = [[]] * length
kg_retail_copper_out[0]  = kg_retail_copper.transpose()
kg_retail_copper_out[1]  = kg_retail_copper_i.transpose()
kg_retail_copper_out[2]  = kg_retail_copper_o.transpose()
for item in range(0,length):
    kg_retail_copper_out[item].insert(0,'material', ['copper'] * 26)
    kg_retail_copper_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_copper_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_copper_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_aluminium_out  = [[]] * length
kg_retail_aluminium_out[0]  = kg_retail_aluminium.transpose()
kg_retail_aluminium_out[1]  = kg_retail_aluminium_i.transpose()
kg_retail_aluminium_out[2]  = kg_retail_aluminium_o.transpose()
for item in range(0,length):
    kg_retail_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
    kg_retail_aluminium_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_aluminium_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_retail_glass_out  = [[]] * length
kg_retail_glass_out[0]  = kg_retail_glass.transpose()
kg_retail_glass_out[1]  = kg_retail_glass_i.transpose()
kg_retail_glass_out[2]  = kg_retail_glass_o.transpose()
for item in range(0,length):
    kg_retail_glass_out[item].insert(0,'material', ['glass'] * 26)
    kg_retail_glass_out[item].insert(0,'area', ['commercial'] * 26)
    kg_retail_glass_out[item].insert(0,'type', ['retail'] * 26)
    kg_retail_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# hotels & restaurants

kg_hotels_steel_out  = [[]] * length
kg_hotels_steel_out[0]  = kg_hotels_steel.transpose()
kg_hotels_steel_out[1]  = kg_hotels_steel_i.transpose()
kg_hotels_steel_out[2]  = kg_hotels_steel_o.transpose()
for item in range(0,length):
    kg_hotels_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_hotels_steel_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_steel_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_steel_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_cement_out  = [[]] * length
kg_hotels_cement_out[0]  = kg_hotels_cement.transpose()
kg_hotels_cement_out[1]  = kg_hotels_cement_i.transpose()
kg_hotels_cement_out[2]  = kg_hotels_cement_o.transpose()
for item in range(0,length):
    kg_hotels_cement_out[item].insert(0,'material', ['cement'] * 26)
    kg_hotels_cement_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_cement_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_cement_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_concrete_out  = [[]] * length
kg_hotels_concrete_out[0]  = kg_hotels_concrete.transpose()
kg_hotels_concrete_out[1]  = kg_hotels_concrete_i.transpose()
kg_hotels_concrete_out[2]  = kg_hotels_concrete_o.transpose()
for item in range(0,length):
    kg_hotels_concrete_out[item].insert(0,'material', ['concrete'] * 26)
    kg_hotels_concrete_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_concrete_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_wood_out  = [[]] * length
kg_hotels_wood_out[0]  = kg_hotels_wood.transpose()
kg_hotels_wood_out[1]  = kg_hotels_wood_i.transpose()
kg_hotels_wood_out[2]  = kg_hotels_wood_o.transpose()
for item in range(0,length):
    kg_hotels_wood_out[item].insert(0,'material', ['wood'] * 26)
    kg_hotels_wood_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_wood_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_wood_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_copper_out  = [[]] * length
kg_hotels_copper_out[0]  = kg_hotels_copper.transpose()
kg_hotels_copper_out[1]  = kg_hotels_copper_i.transpose()
kg_hotels_copper_out[2]  = kg_hotels_copper_o.transpose()
for item in range(0,length):
    kg_hotels_copper_out[item].insert(0,'material', ['copper'] * 26)
    kg_hotels_copper_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_copper_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_copper_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_aluminium_out  = [[]] * length
kg_hotels_aluminium_out[0]  = kg_hotels_aluminium.transpose()
kg_hotels_aluminium_out[1]  = kg_hotels_aluminium_i.transpose()
kg_hotels_aluminium_out[2]  = kg_hotels_aluminium_o.transpose()
for item in range(0,length):
    kg_hotels_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
    kg_hotels_aluminium_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_aluminium_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_hotels_glass_out  = [[]] * length
kg_hotels_glass_out[0]  = kg_hotels_glass.transpose()
kg_hotels_glass_out[1]  = kg_hotels_glass_i.transpose()
kg_hotels_glass_out[2]  = kg_hotels_glass_o.transpose()
for item in range(0,length):
    kg_hotels_glass_out[item].insert(0,'material', ['glass'] * 26)
    kg_hotels_glass_out[item].insert(0,'area', ['commercial'] * 26)
    kg_hotels_glass_out[item].insert(0,'type', ['hotels'] * 26)
    kg_hotels_glass_out[item].insert(0,'flow', [tag[item]] * 26)

# government (schools, government, public transport, hospitals)
kg_govern_steel_out  = [[]] * length
kg_govern_steel_out[0]  = kg_govern_steel.transpose()
kg_govern_steel_out[1]  = kg_govern_steel_i.transpose()
kg_govern_steel_out[2]  = kg_govern_steel_o.transpose()
for item in range(0,length):
    kg_govern_steel_out[item].insert(0,'material', ['steel'] * 26)
    kg_govern_steel_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_steel_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_steel_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_cement_out  = [[]] * length
kg_govern_cement_out[0]  = kg_govern_cement.transpose()
kg_govern_cement_out[1]  = kg_govern_cement_i.transpose()
kg_govern_cement_out[2]  = kg_govern_cement_o.transpose()
for item in range(0,length):
    kg_govern_cement_out[item].insert(0,'material', ['cement'] * 26)
    kg_govern_cement_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_cement_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_cement_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_concrete_out  = [[]] * length
kg_govern_concrete_out[0]  = kg_govern_concrete.transpose()
kg_govern_concrete_out[1]  = kg_govern_concrete_i.transpose()
kg_govern_concrete_out[2]  = kg_govern_concrete_o.transpose()
for item in range(0,length):
    kg_govern_concrete_out[item].insert(0,'material', ['concrete'] * 26)
    kg_govern_concrete_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_concrete_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_concrete_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_wood_out  = [[]] * length
kg_govern_wood_out[0]  = kg_govern_wood.transpose()
kg_govern_wood_out[1]  = kg_govern_wood_i.transpose()
kg_govern_wood_out[2]  = kg_govern_wood_o.transpose()
for item in range(0,length):
    kg_govern_wood_out[item].insert(0,'material', ['wood'] * 26)
    kg_govern_wood_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_wood_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_wood_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_copper_out  = [[]] * length
kg_govern_copper_out[0]  = kg_govern_copper.transpose()
kg_govern_copper_out[1]  = kg_govern_copper_i.transpose()
kg_govern_copper_out[2]  = kg_govern_copper_o.transpose()
for item in range(0,length):
    kg_govern_copper_out[item].insert(0,'material', ['copper'] * 26)
    kg_govern_copper_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_copper_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_copper_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_aluminium_out  = [[]] * length
kg_govern_aluminium_out[0]  = kg_govern_aluminium.transpose()
kg_govern_aluminium_out[1]  = kg_govern_aluminium_i.transpose()
kg_govern_aluminium_out[2]  = kg_govern_aluminium_o.transpose()
for item in range(0,length):
    kg_govern_aluminium_out[item].insert(0,'material', ['aluminium'] * 26)
    kg_govern_aluminium_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_aluminium_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_aluminium_out[item].insert(0,'flow', [tag[item]] * 26)

kg_govern_glass_out  = [[]] * length
kg_govern_glass_out[0]  = kg_govern_glass.transpose()
kg_govern_glass_out[1]  = kg_govern_glass_i.transpose()
kg_govern_glass_out[2]  = kg_govern_glass_o.transpose()
for item in range(0,length):
    kg_govern_glass_out[item].insert(0,'material', ['glass'] * 26)
    kg_govern_glass_out[item].insert(0,'area', ['commercial'] * 26)
    kg_govern_glass_out[item].insert(0,'type', ['govern'] * 26)
    kg_govern_glass_out[item].insert(0,'flow', [tag[item]] * 26) 


# stack into 1 dataframe
frames =    [kg_det_rur_steel_out[0], kg_det_rur_cement_out[0], kg_det_rur_concrete_out[0], kg_det_rur_wood_out[0], kg_det_rur_copper_out[0], kg_det_rur_aluminium_out[0], kg_det_rur_glass_out[0],    
             kg_sem_rur_steel_out[0], kg_sem_rur_cement_out[0], kg_sem_rur_concrete_out[0], kg_sem_rur_wood_out[0], kg_sem_rur_copper_out[0], kg_sem_rur_aluminium_out[0], kg_sem_rur_glass_out[0],    
             kg_app_rur_steel_out[0], kg_app_rur_cement_out[0], kg_app_rur_concrete_out[0], kg_app_rur_wood_out[0], kg_app_rur_copper_out[0], kg_app_rur_aluminium_out[0], kg_app_rur_glass_out[0],    
             kg_hig_rur_steel_out[0], kg_hig_rur_cement_out[0], kg_hig_rur_concrete_out[0], kg_hig_rur_wood_out[0], kg_hig_rur_copper_out[0], kg_hig_rur_aluminium_out[0], kg_hig_rur_glass_out[0],    
             kg_det_urb_steel_out[0], kg_det_urb_cement_out[0], kg_det_urb_concrete_out[0], kg_det_urb_wood_out[0], kg_det_urb_copper_out[0], kg_det_urb_aluminium_out[0], kg_det_urb_glass_out[0],    
             kg_sem_urb_steel_out[0], kg_sem_urb_cement_out[0], kg_sem_urb_concrete_out[0], kg_sem_urb_wood_out[0], kg_sem_urb_copper_out[0], kg_sem_urb_aluminium_out[0], kg_sem_urb_glass_out[0],    
             kg_app_urb_steel_out[0], kg_app_urb_cement_out[0], kg_app_urb_concrete_out[0], kg_app_urb_wood_out[0], kg_app_urb_copper_out[0], kg_app_urb_aluminium_out[0], kg_app_urb_glass_out[0],    
             kg_hig_urb_steel_out[0], kg_hig_urb_cement_out[0], kg_hig_urb_concrete_out[0], kg_hig_urb_wood_out[0], kg_hig_urb_copper_out[0], kg_hig_urb_aluminium_out[0], kg_hig_urb_glass_out[0],
             kg_office_steel_out[0],  kg_office_cement_out[0],  kg_office_concrete_out[0],  kg_office_wood_out[0],  kg_office_copper_out[0],  kg_office_aluminium_out[0],  kg_office_glass_out[0],
             kg_retail_steel_out[0],  kg_retail_cement_out[0],  kg_retail_concrete_out[0],  kg_retail_wood_out[0],  kg_retail_copper_out[0],  kg_retail_aluminium_out[0],  kg_retail_glass_out[0],
             kg_hotels_steel_out[0],  kg_hotels_cement_out[0],  kg_hotels_concrete_out[0],  kg_hotels_wood_out[0],  kg_hotels_copper_out[0],  kg_hotels_aluminium_out[0],  kg_hotels_glass_out[0],
             kg_govern_steel_out[0],  kg_govern_cement_out[0],  kg_govern_concrete_out[0],  kg_govern_wood_out[0],  kg_govern_copper_out[0],  kg_govern_aluminium_out[0],  kg_govern_glass_out[0],
                                                           
             kg_det_rur_steel_out[1], kg_det_rur_cement_out[1], kg_det_rur_concrete_out[1], kg_det_rur_wood_out[1], kg_det_rur_copper_out[1], kg_det_rur_aluminium_out[1], kg_det_rur_glass_out[1],    
             kg_sem_rur_steel_out[1], kg_sem_rur_cement_out[1], kg_sem_rur_concrete_out[1], kg_sem_rur_wood_out[1], kg_sem_rur_copper_out[1], kg_sem_rur_aluminium_out[1], kg_sem_rur_glass_out[1],    
             kg_app_rur_steel_out[1], kg_app_rur_cement_out[1], kg_app_rur_concrete_out[1], kg_app_rur_wood_out[1], kg_app_rur_copper_out[1], kg_app_rur_aluminium_out[1], kg_app_rur_glass_out[1],    
             kg_hig_rur_steel_out[1], kg_hig_rur_cement_out[1], kg_hig_rur_concrete_out[1], kg_hig_rur_wood_out[1], kg_hig_rur_copper_out[1], kg_hig_rur_aluminium_out[1], kg_hig_rur_glass_out[1],    
             kg_det_urb_steel_out[1], kg_det_urb_cement_out[1], kg_det_urb_concrete_out[1], kg_det_urb_wood_out[1], kg_det_urb_copper_out[1], kg_det_urb_aluminium_out[1], kg_det_urb_glass_out[1],    
             kg_sem_urb_steel_out[1], kg_sem_urb_cement_out[1], kg_sem_urb_concrete_out[1], kg_sem_urb_wood_out[1], kg_sem_urb_copper_out[1], kg_sem_urb_aluminium_out[1], kg_sem_urb_glass_out[1],    
             kg_app_urb_steel_out[1], kg_app_urb_cement_out[1], kg_app_urb_concrete_out[1], kg_app_urb_wood_out[1], kg_app_urb_copper_out[1], kg_app_urb_aluminium_out[1], kg_app_urb_glass_out[1],    
             kg_hig_urb_steel_out[1], kg_hig_urb_cement_out[1], kg_hig_urb_concrete_out[1], kg_hig_urb_wood_out[1], kg_hig_urb_copper_out[1], kg_hig_urb_aluminium_out[1], kg_hig_urb_glass_out[1],
             kg_office_steel_out[1],  kg_office_cement_out[1],  kg_office_concrete_out[1],  kg_office_wood_out[1],  kg_office_copper_out[1],  kg_office_aluminium_out[1],  kg_office_glass_out[1],
             kg_retail_steel_out[1],  kg_retail_cement_out[1],  kg_retail_concrete_out[1],  kg_retail_wood_out[1],  kg_retail_copper_out[1],  kg_retail_aluminium_out[1],  kg_retail_glass_out[1],
             kg_hotels_steel_out[1],  kg_hotels_cement_out[1],  kg_hotels_concrete_out[1],  kg_hotels_wood_out[1],  kg_hotels_copper_out[1],  kg_hotels_aluminium_out[1],  kg_hotels_glass_out[1],
             kg_govern_steel_out[1],  kg_govern_cement_out[1],  kg_govern_concrete_out[1],  kg_govern_wood_out[1],  kg_govern_copper_out[1],  kg_govern_aluminium_out[1],  kg_govern_glass_out[1],
            
             kg_det_rur_steel_out[2], kg_det_rur_cement_out[2], kg_det_rur_concrete_out[2], kg_det_rur_wood_out[2], kg_det_rur_copper_out[2], kg_det_rur_aluminium_out[2], kg_det_rur_glass_out[2],    
             kg_sem_rur_steel_out[2], kg_sem_rur_cement_out[2], kg_sem_rur_concrete_out[2], kg_sem_rur_wood_out[2], kg_sem_rur_copper_out[2], kg_sem_rur_aluminium_out[2], kg_sem_rur_glass_out[2],    
             kg_app_rur_steel_out[2], kg_app_rur_cement_out[2], kg_app_rur_concrete_out[2], kg_app_rur_wood_out[2], kg_app_rur_copper_out[2], kg_app_rur_aluminium_out[2], kg_app_rur_glass_out[2],    
             kg_hig_rur_steel_out[2], kg_hig_rur_cement_out[2], kg_hig_rur_concrete_out[2], kg_hig_rur_wood_out[2], kg_hig_rur_copper_out[2], kg_hig_rur_aluminium_out[2], kg_hig_rur_glass_out[2],    
             kg_det_urb_steel_out[2], kg_det_urb_cement_out[2], kg_det_urb_concrete_out[2], kg_det_urb_wood_out[2], kg_det_urb_copper_out[2], kg_det_urb_aluminium_out[2], kg_det_urb_glass_out[2],    
             kg_sem_urb_steel_out[2], kg_sem_urb_cement_out[2], kg_sem_urb_concrete_out[2], kg_sem_urb_wood_out[2], kg_sem_urb_copper_out[2], kg_sem_urb_aluminium_out[2], kg_sem_urb_glass_out[2],    
             kg_app_urb_steel_out[2], kg_app_urb_cement_out[2], kg_app_urb_concrete_out[2], kg_app_urb_wood_out[2], kg_app_urb_copper_out[2], kg_app_urb_aluminium_out[2], kg_app_urb_glass_out[2],    
             kg_hig_urb_steel_out[2], kg_hig_urb_cement_out[2], kg_hig_urb_concrete_out[2], kg_hig_urb_wood_out[2], kg_hig_urb_copper_out[2], kg_hig_urb_aluminium_out[2], kg_hig_urb_glass_out[2], 
             kg_office_steel_out[2],  kg_office_cement_out[2],  kg_office_concrete_out[2],  kg_office_wood_out[2],  kg_office_copper_out[2],  kg_office_aluminium_out[2],  kg_office_glass_out[2],
             kg_retail_steel_out[2],  kg_retail_cement_out[2],  kg_retail_concrete_out[2],  kg_retail_wood_out[2],  kg_retail_copper_out[2],  kg_retail_aluminium_out[2],  kg_retail_glass_out[2],
             kg_hotels_steel_out[2],  kg_hotels_cement_out[2],  kg_hotels_concrete_out[2],  kg_hotels_wood_out[2],  kg_hotels_copper_out[2],  kg_hotels_aluminium_out[2],  kg_hotels_glass_out[2],
             kg_govern_steel_out[2],  kg_govern_cement_out[2],  kg_govern_concrete_out[2],  kg_govern_wood_out[2],  kg_govern_copper_out[2],  kg_govern_aluminium_out[2],  kg_govern_glass_out[2]   ]

material_output = pd.concat(frames)
material_output.to_csv('output\\material_output.csv')

# SQUARE METERS (Stock results) ---------------------------------------------------

length = 3
tag = ['stock', 'inflow', 'outflow']

m2_det_rur_out  = [[]] * length
m2_det_rur_out[0]  = m2_det_rur.transpose()
m2_det_rur_out[1]  = m2_det_rur_i.transpose()
m2_det_rur_out[2]  = m2_det_rur_o.transpose()
for item in range(0,length):
    m2_det_rur_out[item].insert(0,'area', ['rural'] * 26)
    m2_det_rur_out[item].insert(0,'type', ['detached'] * 26)
    m2_det_rur_out[item].insert(0,'flow', [tag[item]] * 26)

m2_sem_rur_out  = [[]] * length
m2_sem_rur_out[0]  = m2_sem_rur.transpose()
m2_sem_rur_out[1]  = m2_sem_rur_i.transpose()
m2_sem_rur_out[2]  = m2_sem_rur_o.transpose()
for item in range(0,length):
    m2_sem_rur_out[item].insert(0,'area', ['rural'] * 26)
    m2_sem_rur_out[item].insert(0,'type', ['semi-detached'] * 26)
    m2_sem_rur_out[item].insert(0,'flow', [tag[item]] * 26)

m2_app_rur_out  = [[]] * length
m2_app_rur_out[0]  = m2_app_rur.transpose()
m2_app_rur_out[1]  = m2_app_rur_i.transpose()
m2_app_rur_out[2]  = m2_app_rur_o.transpose()
for item in range(0,length):
    m2_app_rur_out[item].insert(0,'area', ['rural'] * 26)
    m2_app_rur_out[item].insert(0,'type', ['appartments'] * 26)
    m2_app_rur_out[item].insert(0,'flow', [tag[item]] * 26)

m2_hig_rur_out  = [[]] * length
m2_hig_rur_out[0]  = m2_hig_rur.transpose()
m2_hig_rur_out[1]  = m2_hig_rur_i.transpose()
m2_hig_rur_out[2]  = m2_hig_rur_o.transpose()  
for item in range(0,length):  
    m2_hig_rur_out[item].insert(0,'area', ['rural'] * 26)
    m2_hig_rur_out[item].insert(0,'type', ['high-rise'] * 26)
    m2_hig_rur_out[item].insert(0,'flow', [tag[item]] * 26)

m2_det_urb_out  = [[]] * length
m2_det_urb_out[0]  = m2_det_urb.transpose()
m2_det_urb_out[1]  = m2_det_urb_i.transpose()
m2_det_urb_out[2]  = m2_det_urb_o.transpose()  
for item in range(0,length):  
    m2_det_urb_out[item].insert(0,'area', ['urban'] * 26)
    m2_det_urb_out[item].insert(0,'type', ['detached'] * 26)
    m2_det_urb_out[item].insert(0,'flow', [tag[item]] * 26)

m2_sem_urb_out  = [[]] * length
m2_sem_urb_out[0]  = m2_sem_urb.transpose()
m2_sem_urb_out[1]  = m2_sem_urb_i.transpose()
m2_sem_urb_out[2]  = m2_sem_urb_o.transpose()  
for item in range(0,length):  
    m2_sem_urb_out[item].insert(0,'area', ['urban'] * 26)
    m2_sem_urb_out[item].insert(0,'type', ['semi-detached'] * 26)
    m2_sem_urb_out[item].insert(0,'flow', [tag[item]] * 26)
    
m2_app_urb_out  = [[]] * length
m2_app_urb_out[0]  = m2_app_urb.transpose()
m2_app_urb_out[1]  = m2_app_urb_i.transpose()
m2_app_urb_out[2]  = m2_app_urb_o.transpose()  
for item in range(0,length):  
    m2_app_urb_out[item].insert(0,'area', ['urban'] * 26)
    m2_app_urb_out[item].insert(0,'type', ['appartments'] * 26)
    m2_app_urb_out[item].insert(0,'flow', [tag[item]] * 26)
    
m2_hig_urb_out  = [[]] * length
m2_hig_urb_out[0]  = m2_hig_urb.transpose()
m2_hig_urb_out[1]  = m2_hig_urb_i.transpose()
m2_hig_urb_out[2]  = m2_hig_urb_o.transpose()  
for item in range(0,length):  
    m2_hig_urb_out[item].insert(0,'area', ['urban'] * 26)
    m2_hig_urb_out[item].insert(0,'type', ['high-rise'] * 26)
    m2_hig_urb_out[item].insert(0,'flow', [tag[item]] * 26)

# COMMERCIAL
m2_office_out  = [[]] * length
m2_office_out[0]  = commercial_m2_office.transpose()
m2_office_out[1]  = m2_office_i.transpose()
m2_office_out[2]  = m2_office_o.transpose()
for item in range(0,length):
    m2_office_out[item].insert(0,'area', ['commercial'] * 26)
    m2_office_out[item].insert(0,'type', ['office'] * 26)
    m2_office_out[item].insert(0,'flow', [tag[item]] * 26)

m2_retail_out  = [[]] * length
m2_retail_out[0]  = commercial_m2_retail.transpose()
m2_retail_out[1]  = m2_retail_i.transpose()
m2_retail_out[2]  = m2_retail_o.transpose()
for item in range(0,length):
    m2_retail_out[item].insert(0,'area', ['commercial'] * 26)
    m2_retail_out[item].insert(0,'type', ['retail'] * 26)
    m2_retail_out[item].insert(0,'flow', [tag[item]] * 26)

m2_hotels_out  = [[]] * length
m2_hotels_out[0]  = commercial_m2_hotels.transpose()
m2_hotels_out[1]  = m2_hotels_i.transpose()
m2_hotels_out[2]  = m2_hotels_o.transpose()
for item in range(0,length):
    m2_hotels_out[item].insert(0,'area', ['commercial'] * 26)
    m2_hotels_out[item].insert(0,'type', ['hotels'] * 26)
    m2_hotels_out[item].insert(0,'flow', [tag[item]] * 26)

m2_govern_out  = [[]] * length
m2_govern_out[0]  = commercial_m2_govern.transpose()
m2_govern_out[1]  = m2_govern_i.transpose()
m2_govern_out[2]  = m2_govern_o.transpose()
for item in range(0,length):
    m2_govern_out[item].insert(0,'area', ['commercial'] * 26)
    m2_govern_out[item].insert(0,'type', ['govern'] * 26)
    m2_govern_out[item].insert(0,'flow', [tag[item]] * 26)

frames2 = [m2_det_rur_out[0], m2_sem_rur_out[0], m2_app_rur_out[0], m2_hig_rur_out[0], m2_det_urb_out[0], m2_sem_urb_out[0], m2_app_urb_out[0], m2_hig_urb_out[0],
           m2_office_out[0],  m2_retail_out[0],  m2_hotels_out[0],  m2_govern_out[0],
           m2_det_rur_out[1], m2_sem_rur_out[1], m2_app_rur_out[1], m2_hig_rur_out[1], m2_det_urb_out[1], m2_sem_urb_out[1], m2_app_urb_out[1], m2_hig_urb_out[1],
           m2_office_out[1],  m2_retail_out[1],  m2_hotels_out[1],  m2_govern_out[1],
           m2_det_rur_out[2], m2_sem_rur_out[2], m2_app_rur_out[2], m2_hig_rur_out[2], m2_det_urb_out[2], m2_sem_urb_out[2], m2_app_urb_out[2], m2_hig_urb_out[2],
           m2_office_out[2],  m2_retail_out[2],  m2_hotels_out[2],  m2_govern_out[2] ]

sqmeters_output = pd.concat(frames2)
sqmeters_output.to_csv('output\\sqmeters_output.csv') # in kt

