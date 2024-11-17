import sys

from epics import PV, caget, cainfo, camonitor, caput
import os

# current_file_path = os.path.abspath(__file__)  # \clapa\sim_epics.py
# current_dir = os.path.dirname(current_file_path)  # \clapa
# pku_dir = os.path.dirname(current_dir)  # pku
# pybeamsim_dir = pku_dir
# sys.path.append(pku_dir)

import pyBeamSim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# create channel by dictionary
# pv_channels = {
#     'a': PV('quad2:gradient'),
#     'b': PV('quad1:gradient')
# }
# print(pv_channels['a'].get(), pv_channels['b'].get(), '\n')
# print(pv_channels['a'].get())

def simulate_by_seq(value_list):
    ref_energy = 5
    num_particle = 102400
    simulator = pyBeamSim.BeamSimulator()
    simulator.init_beam(num_particle, 938.272046, 1.0, 0.0)
    simulator.set_beamTwiss(0, 0.003, 0.0001, 0, 0.003, 0.0001, 0, 8, 3.1415926e-11, 0, ref_energy, 500, 1)
    simulator.save_initial_beam()
    simulator.UpdateBeamParameters()
    print(f'before simulate: {simulator.getAvgX()}')
    simulator.add_Quad('quad1', value_list[0], 0.012, value_list[1])
    simulator.add_Quad('quad2', value_list[2], 0.025, value_list[3])
    simulator.simulate_all()
    simulator.UpdateBeamParameters()
    print(f'after simulate: {simulator.getAvgX()}')


pv_list = ['quad1:length', 'quad1:gradient', 'quad2:length', 'quad2:gradient']
pv_channels = {pv: PV(pv) for pv in pv_list}
pv_values = [pv_channels[pv].get() for pv in pv_list]
simulate_by_seq(pv_values)