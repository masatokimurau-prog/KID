from resonator_tools import circuit

import ds_style

port1 = circuit.notch_port()
#port1.add_froms2p('../daqpc_copy/s2p_20251113/1_3.5K.s2p',3,4,'dBmagphasedeg',fdata_unit=1e0,delimiter=None)
port1.add_froms2p('../daqpc_copy/s2p_20251113/2_4.74K.s2p',3,4,'dBmagphasedeg',fdata_unit=1e0,delimiter=None)
#port1.autofit()
port1.GUIfit()
print("Fit results:", port1.fitresults)
port1.plotall()
print("single photon limit:", port1.get_single_photon_limit(diacorr=True), "dBm")
print("photons in reso for input -140dBm:", port1.get_photons_in_resonator(-140,unit='dBm',diacorr=True), "photons")
print("done")

