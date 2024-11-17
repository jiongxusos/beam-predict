import pyBeamSim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import accumulate



# initial the beam
ref_energy = 5
num_particle = 1024000
simulator = pyBeamSim.BeamSimulator()
simulator.init_beam(num_particle, 938.272046, 1.0, 0.0)
simulator.set_beamTwiss(0, 0.003, 0.0001, 0, 0.003, 0.0001, 0, 8, 3.1415926e-11, 0, ref_energy, 500, 1)
simulator.save_initial_beam()

# initial the beamline
simulator.load_Beamline_From_DatFile("../tracewin_data/clapa2.dat", 5)
elements = simulator.get_Beamline_ElementNames()
lengths = simulator.get_Beamline_ElementLengths()    # /m
apertures = list(simulator.get_Beamline_ElementApertures())  # /m , length: 70
envelope = simulator.simulate_and_getEnvelope()
simulator.UpdateBeamParameters()
print(len(apertures))
# beam_size along z, x_sig and y_sig here
x_avg = envelope['Avg_x']
x_sig = envelope['Sig_x']  # length: 71
y_sig = envelope['Sig_y']
z = list(accumulate(lengths))   # length: 70
print(z)
print(elements)
z.insert(0, 0)

# plot x_sig along z with aperture
fig, ax1 = plt.subplots()
ax1.set_title('x_sig along z with aperture')
ax1.set_xlabel('z/m')
ax1.set_ylabel('x_sig/m', color='tab:blue')
ax1.plot(z, x_avg, color='tab:blue', label='x_sig')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# use plot to plot aperture
ax2 = ax1.twinx()
ax2.set_ylabel('aperture/m', color='tab:orange')

# use the same tick for two y_axis
# y_min = min(min(x_sig), min(apertures))
# y_max = max(max(x_sig), max(apertures))
# ax1.set_ylim(y_min, y_max)
# ax2.set_ylim(y_min, y_max)

# ax1.set_yticks(range(int(y_min), int(y_max) + 1))
# ax2.set_yticks(range(int(y_min), int(y_max) + 1))

# ax1.set_ylim(0, 0.05)

# avoid the bound
apertures.insert(-1, apertures[-1])
for i in range(len(z) - 1):
    ax2.plot([z[i], z[i + 1]], [apertures[i], apertures[i]], color='tab:orange')
    ax2.plot([z[i + 1], z[i + 1]], [apertures[i], apertures[i + 1]], color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.show()


# print(len(elements), len(lengths), len(apertures))   70 70 70
"""
print(elements)
['DRIFT1', 'S1_low_1focus', 'DRIFT2', 'DRIFT3', 'DRIFT4', 'DRIFT5', 'DRIFT6', 'DRIFT7', 
 'S2_low_1focus', 'DRIFT8', 'DRIFT9', 'DRIFT10', 'S3_low_1focus', 'DRIFT11', 'DRIFT12', 
 'DRIFT13', 'DRIFT14', 'DRIFT15', 'DRIFT16', 'DRIFT17', 'DRIFT18', 'DRIFT19', 'DRIFT20', 
 'DRIFT21', 'DRIFT22', 'Q1', 'DRIFT23', 'Q2', 'DRIFT24', 'BEND1', 'DRIFT25', 'Q3_LB', 'DRIFT26',
 'DRIFT27', 'DRIFT28', 'DRIFT29', 'Q4_LB', 'DRIFT30', 'BEND2', 'DRIFT31', 'DRIFT32',
 'Q5_0.0015err', 'DRIFT33', 'Q6_0.0015err', 'DRIFT34', 'OCTx_off', 'DRIFT35', 'Q7', 
 'DRIFT36', 'Q8', 'DRIFT37', 'OCTy_off', 'DRIFT38', 'Qj6_long_d12', 'DRIFT39', 'Qj7_long_d12',
 'DRIFT40', 'Qj8_long_d12', 'DRIFT41', 'DRIFT42', 'DRIFT43', 'DRIFT44', 'DRIFT45', 'DRIFT46',
 'DRIFT47', 'DRIFT48', 'DRIFT49', 'DRIFT50', 'DRIFT51', 'DRIFT52']
 """

# count all apertures
"""
for the .dat   stod: string to double

drift  length: 340mm, aperture: 50mm
solenoid
s1_long: solenoid
s1_low_1foucs: solenoid
bend:   

        0, angle: abs(stod(45) * RADIAN), radius: 1000mm, fieldindex: 0, aperture: stod(30)mm
        
quad    length: 150mm, gradient: 90901841 T/m, aperture: 36mm  
"""