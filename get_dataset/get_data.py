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
    
    # radius: 0.65,
    # angle: 45 *pi/180 =0.7854,
    # HalfGap: 70/1000/2 = 0.035
    # EdgeAngleIn: 0*pi/180 = 0
    # EdgeAngleOut: 0 * pi/180 = 0
    # fieldindex: 0
    # Aperture: 0.03
    # KineticEnergy:ReferenceEnergy = 5
    # Length: radius * angel = 0.511

APERTURERECTANGULAR:  XLeft = 10/1000, XRight = 10/1000, YBottom = 100/1000, YTop = 1000/1000
!APERTURERECTANGULAR 10 10 100 1000

"""

param_name = ['quad1_length', 'quad1_gradient', 'quad2_length', 'quad2_gradient', 'quad3_length',
             'quad3_gradient', 'quad4_length', 'quad4_gradient', 'quad5_length', 'quad5_gradient']
for i in range(100):

    simulator.restore_initial_beam()
    # key
    simulator.UpdateBeamParameters()

    param = {}
    for j in range(5):
        param[f'quad{j+1}_length'] = np.random.uniform(0.1, 0.5)
        param[f'quad{j+1}_gradient'] = np.random.uniform(-20, 20)

    param_list.append(dict(zip(param_name, list(param.values()))))

    """beamline initial """

    simulator.init_Beamline()  #needed without interface

    simulator.add_Drift('drift1', 0.19, 0.05)
    simulator.add_Quad('quad1', param['quad1_length'], 0.012, param['quad1_gradient'])
    simulator.add_Drift('drift2', 0.058, 0.05)
    simulator.add_Quad('quad2', param['quad2_length'], 0.028, param['quad2_gradient'])
    simulator.add_Drift('drift3', 0.058, 0.05)
    simulator.add_Quad('quad3', param['quad3_length'], 0.028, param['quad3_gradient'])
    simulator.add_Drift('drift4', 3.365, 0.05)
    # simulator.add_Bend('bend1', 0.511, 0.03, 0.7854, 0, 0,
    #                    0, 1.0, 5)
    simulator.add_Drift('drift5', 1.365, 0.05)
    # simulator.add_ApertureRectangular('aper1', 0.01, 0.01, 0.1, 0.1)
    simulator.add_Drift('drift6', 0.2, 0.05)
    simulator.add_Quad('quad4', param['quad4_length'], 0.028, param['quad4_gradient'])
    simulator.add_Drift('drift7', 0.2, 0.05)
    simulator.add_Quad('quad5', param['quad5_length'], 0.028, param['quad5_gradient'])
    simulator.add_Drift('drift8', 1.015, 0.05)


    # generate the dataframe
    envelope = simulator.simulate_and_getEnvelope(use_spacecharge=True)

    x_avg = envelope['Avg_x']
    x_sig = envelope['Sig_x']
    y_sig = envelope['Sig_y']

    # need a function to get twiss params
    x_avg0, x_avg1, x_avg2, x_avg3, x_avg4 = x_avg[0], x_avg[2], x_avg[4], x_avg[10], x_avg[13]

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
df.to_csv('data/data_10000.csv')

df_sig4 = pd.DataFrame(result_sig4)
df_mlp = pd.concat((df_param, df_sig4), axis=1)
df_mlp.to_csv('data/attention.csv')

"""simulator.load_Beamline_From_DatFile("./tracewin_data/clapa1_5MeV.dat", 5)
names_list = simulator.get_Beamline_ElementNames()
print(names_list)
len_list = simulator.get_Beamline_ElementLengths()
print(len(len_list))"""