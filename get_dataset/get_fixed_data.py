import numpy as np
import pyBeamSim
import pandas as pd

"""
clapa1 

drift - Q1 - drift - Q2 - drift - Q3 - drift - Bend - drift - APERTURERECTANGULAR - drift - Q4 - drift - Q5 - drift 

there is no beamline interface, so change .dat instead

reserve the drift space, quad length etc. just change the values of reality
 """
ref_energy = 5
num_particle = 102400
simulator = pyBeamSim.BeamSimulator()
# simulator.free_beam()
simulator.init_beam(num_particle, 938.272046, 1.0, 0.0)
simulator.set_beamTwiss(0, 0.003, 0.0001, 0, 0.003, 0.0001, 0, 8, 3.1415926e-11, 0, ref_energy, 500, 1)
simulator.save_initial_beam()

simulator.init_spacecharge()

"""" parameters random """
param_list = []
result_list = []
result_sig4 = []

"""  tracewin_data:mm   Hpsim:m   stod: string to double
upper: Hpsim
lower: tracewin_data

quad :  length = 100/1000, aperture = 12/1000, field_gradient = 13.19614669
Q1:quad 100 13.19614669 12

dipole(EDGE-BEND-EDGE) :    RADIAN + stod + abs 
dipole(EDGE-BEND-EDGE) :  Radius = , Angle = , HalfGap = , EdgeAngleIn = , EdgeAngleOut = 
EDGE 0 650 70 0.45 2.8 40 0
BEND 45 650 0 30 0
EDGE 0 650 70 0.45 2.8 40 0

APERTURERECTANGULAR:  XLeft = 10/1000, XRight = 10/1000, YBottom = 100/1000, YTop = 1000/1000
!APERTURERECTANGULAR 10 10 100 1000

"""

param_name = ['quad1']
for i in range(1000):

    simulator.restore_initial_beam()
    # key
    simulator.UpdateBeamParameters()

    drift1_length = 0.3
    drift1_aperture = 0.05
    quad1_length = 0.4
    quad1_aperture = 0.012
    quad1_field_gradient = 15
    drift2_length = 0.2
    drift2_aperture = 0.05
    quad2_length = 0.4
    quad2_aperture = 0.028
    quad2_field_gradient = 5

    parameters = {
        'drift1_len': drift1_length,
        'quad1_len': quad1_length,
        'quad1_gra': quad1_field_gradient,
        'drift2_len': drift2_length,
        'quad2_len': quad2_length,
        'quad2_gra': quad2_field_gradient
    }

    param_list.append(parameters)

    """beamline initial """


    simulator.init_Beamline()  #needed without interface

    simulator.add_Drift('drift1', drift1_length, drift1_aperture)
    simulator.add_Quad('quad1', quad1_length, quad1_aperture, quad1_field_gradient)
    simulator.add_Drift('drift2', drift2_length, drift2_aperture)
    simulator.add_Quad('quad2', quad2_length, quad2_aperture, quad2_field_gradient)
    # simulator.add_Quad("quad2", 0.1, 0.1, 0.1)
    # simulator.add_Drift('drift3', 0.1, 0.1)
    # simulator.add_Quad("quad3", 0.1, 0.1, 0.1)
    # simulator.add_Drift('drift4', 0.1, 0.1)
    # simulator.add_Bend('Bend1', 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    # simulator.add_Drift('drift5', 0.01, 0.01)
    # simulator.add_ApertureRectangular('rec1', 0.01, 0.01, 0.01, 0.01)
    # simulator.add_Drift('drift6', 0.01, 0.01)
    # simulator.add_Quad("quad4", 0.1, 0.1, 0.1)
    # simulator.add_Drift('drift7', 0.01, 0.01)
    # simulator.add_Quad("quad5", 0.1, 0.1, 0.1)
    # simulator.add_Drift('drift8', 0.01, 0.01)

    """" generate the dataframe"""


    envelope = simulator.simulate_and_getEnvelope(use_spacecharge=True)

    x_avg = envelope['Avg_x']
    x_sig = envelope['Sig_x']
    y_sig = envelope['Sig_y']

    # need a function to get twiss params
    x_avg0, x_avg1, x_avg2, x_avg3, x_avg4 = x_avg[:5]

    x_avg_results = {
        'x_avg0': x_avg0,
        'x_avg1': x_avg1,
        'x_avg2': x_avg2,
        'x_avg3': x_avg3,
        'x_avg4': x_avg4
    }

    sig_results = {
        'x_sig4': x_sig[4],
        'y_sig4': y_sig[4]
    }

    result_list.append(x_avg_results)
    result_sig4.append(sig_results)

df_param = pd.DataFrame(param_list)
df_result = pd.DataFrame(result_list)
df = pd.concat((df_param, df_result), axis=1)

print(df)
df.to_csv('data/fixed_input2.csv')

df_sig4 = pd.DataFrame(result_sig4)
df_mlp = pd.concat((df_param, df_sig4), axis=1)
# df_mlp.to_csv('data/fixed_input.csv')

"""simulator.load_Beamline_From_DatFile("./tracewin_data/clapa1_5MeV.dat", 5)
names_list = simulator.get_Beamline_ElementNames()
print(names_list)
len_list = simulator.get_Beamline_ElementLengths()
print(len(len_list))"""