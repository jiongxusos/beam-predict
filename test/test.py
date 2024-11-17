import pyBeamSim

"""
need to make a fun to visit beamline
"""

ref_energy = 5
num_particle = 102400
simulator = pyBeamSim.BeamSimulator()
# simulator.free_beam()
simulator.init_beam(num_particle, 938.272046, 1.0, 0.0)
simulator.set_beamTwiss(0, 0.003, 0.0001, 0, 0.003, 0.0001, 0, 8, 3.1415926e-11, 0, ref_energy, 500, 1)
simulator.save_initial_beam()

simulator.load_Beamline_From_DatFile("./tracewin_data/clapa1_5MeV.dat", 5)

envelope = simulator.simulate_and_getEnvelope_CPU()  # a dict

# length1 = simulator.get_length_index(2)
# simulator.change_length_by_index(2, 0.50)
# length2 = simulator.get_length_by_index(2)
# print(f'before: {length1}, after: {length2}')


# simulator.plot_envelope(envelope)
# simulator.plot_beam_phase_map()
#
# simulator.UpdateBeamParameters_CPU()

"""['DRIFT1', 'DRIFT2', 'Q1', 'DRIFT3', 'Q2', 'DRIFT4', 'Q3', 'DRIFT5', 'DRIFT6', 'DRIFT7', 'DRIFT8',
 'DRIFT9', 'DRIFT10', 'DRIFT11', 'DRIFT12', 'DRIFT13', 'DRIFT14', 'DRIFT15', 'DRIFT16', 'DRIFT17', 
 'DRIFT18', 'DRIFT19', 'DRIFT20', 'DRIFT21', 'DRIFT22', 'DRIFT23', 'DRIFT24', 'DRIFT25', 'DRIFT26', 
 'DRIFT27', 'DRIFT28', 'DRIFT29', 'DRIFT30', 'DRIFT31', 'DRIFT32', 'DRIFT33', 'DRIFT34', 'DRIFT35',
 'DRIFT36', 'DRIFT37', 'DRIFT38', 'DRIFT39', 'DRIFT40', 'DRIFT41', 'DRIFT42', 'DRIFT43', 'DRIFT44', 
 'DRIFT45', 'DRIFT46', 'DRIFT47', 'DRIFT48', 'DRIFT49', 'DRIFT50', 'DRIFT51', 'BEND1', 'DRIFT52',
 'DRIFT53', 'DRIFT54', 'DRIFT55', 'DRIFT56', 'DRIFT57', 'DRIFT58', 'DRIFT59', 'DRIFT60', 'DRIFT61', 
 'DRIFT62', 'DRIFT63', 'DRIFT64', 'DRIFT65', 'DRIFT66', 'DRIFT67', 'DRIFT68', 'DRIFT69', 'DRIFT70', 
 'DRIFT71', 'DRIFT72', 'Q4', 'DRIFT73', 'DRIFT74', 'Q5', 'DRIFT75', 'DRIFT76', 'DRIFT77', 'DRIFT78', 
 'DRIFT79', 'DRIFT80', 'DRIFT81', 'DRIFT82', 'DRIFT83', 'DRIFT84']"""
