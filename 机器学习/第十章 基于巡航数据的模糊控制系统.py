# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 17:26:48 2022

作者：李一邨
人工智能算法案例大全：基于Python
浙大城市学院、杭州伊园科技有限公司
浙江大学 博士
中国民盟盟员
Email:liyicun_yykj@163.com
"""

#%%
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#%%

temperature = ctrl.Antecedent(np.arange(0, 111, 1), 'temperature')
cloud = ctrl.Antecedent(np.arange(0, 101, 1), 'cloud')

speed = ctrl.Consequent(np.arange(0, 101, 1), 'speed')

print('Temperature: ', temperature.universe)
print('Cloud: ', cloud.universe)
print('Speed: ', speed.universe)

#%%

temperature['Freezing'] = fuzz.trapmf(temperature.universe, [0,0,30,50])
temperature['Cool'] = fuzz.trimf(temperature.universe, [30, 50, 70])
temperature['Warm'] = fuzz.trimf(temperature.universe, [50, 70, 90])
temperature['Hot'] = fuzz.trapmf(temperature.universe, [70,90,110,110])
 
temperature.view()

#%%

cloud['Sunny'] = fuzz.trapmf(cloud.universe, [0,0,20,40])
cloud['Cloudy'] = fuzz.trimf(cloud.universe, [20, 50, 80])
cloud['Overcast'] = fuzz.trapmf(cloud.universe, [60,80,100,100])
cloud.view()

#%%


speed['Slow'] = fuzz.trapmf(speed.universe, [0, 0, 25,75])
speed['Fast'] = fuzz.trapmf(speed.universe, [25, 75, 100,100])
speed.view()

#%%
rule1 = ctrl.Rule(temperature['Warm'] & cloud['Sunny'], speed['Fast'] )
rule2 = ctrl.Rule(temperature['Cool'] & cloud['Cloudy'], speed['Slow'])

#%%


Cruise_ctrl = ctrl.ControlSystem([rule1, rule2])
#%%


Cruise = ctrl.ControlSystemSimulation(Cruise_ctrl)

#%%


Cruise.input['temperature'] = 64
Cruise.input['cloud'] = 22

Cruise.compute()


#%%

print('Recommended Speed: ', round(Cruise.output['speed'],3), "miles/hour")
speed.view(sim=Cruise)


#%%

quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')

tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

print('Quality: ', quality.universe)
print('Service: ', service.universe)
print('Tip: ', tip.universe)

#%%

quality.automf(3)
print(quality.terms)
quality.view()

#%%

service.automf(3)
print(service.terms)
service.view()

#%%

tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])
print(tip.terms)
tip.view()

#%%

rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

#%%

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

tipping.input['service'] = 9.8
tipping.input['quality'] = 6.5

tipping.compute()

#%%

print('Recommended Tip: ', round(tipping.output['tip'],1))
tip.view(sim=tipping)

