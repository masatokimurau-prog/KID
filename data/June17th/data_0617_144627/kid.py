
import numpy as np
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(1,1, figsize=(12,6))

import sys
if len(sys.argv)<2:
    print('give the number of waveforms')
    sys.exit()
nwf = int(sys.argv[1])

import niscope
ch0 = []
ch1 = []
deltat = []
with niscope.Session("PXI2Slot2") as session:

    # channel
    # range = 0.01: lsb 0.07mV (min) -> full scale 70mV (pm35mV)
    # range = 0.2: lsb 0.25mV -> full scale 250mV (pm125mV)
    ch0_range = 0.01
    #ch0_range = 0.2
    #session.channels[0].configure_vertical(range=ch0_range, coupling=niscope.VerticalCoupling.DC)
    session.channels[0].configure_vertical(range=ch0_range, coupling=niscope.VerticalCoupling.DC)
    session.channels[0].vertical_offset = 0.0
    ch1_range = 0.01
    ch1_range = 0.01
    #session.channels[1].configure_vertical(range=ch1_range, coupling=niscope.VerticalCoupling.AC)
    session.channels[1].configure_vertical(range=ch1_range, coupling=niscope.VerticalCoupling.DC)
    session.channels[1].vertical_offset = 0.0
    #session.channels[1].vertical_offset = -0.014
    print(session.channels[1].vertical_range, session.channels[1].vertical_offset)
    input_impedance = 50
    #input_impedance = 1e6
    session.channels[0].input_impedance = input_impedance
    session.channels[1].input_impedance = input_impedance

    ch0_bandwidth = -1
    #ch0_bandwidth = 175e6
    session.channels[0].max_input_frequency = ch0_bandwidth
    ch1_bandwidth = -1
    #ch1_bandwidth = 175e6
    session.channels[1].max_input_frequency = ch1_bandwidth

    # horizontal
    sample_rate = 2500e6# max 2.5GS/s for pxie-5160
    time_window = 2e-6# for high temp
    #time_window = 5e-6
    #npts = int(1e5)
    npts = int(sample_rate * time_window)
    ref_position = 20#%
    session.configure_horizontal_timing(min_sample_rate=sample_rate, min_num_pts=npts, ref_position=ref_position, num_records=1, enforce_realtime=True)

    # trigger
    trigger_source = 'VAL_EXTERNAL'
    # trigger_source = '1'
    #trigger_level = 1.6e-3
    trigger_level = 2.2
    #trigger_level = -8e-5
    trigger_slope = niscope.TriggerSlope.POSITIVE
    #trigger_slope = niscope.TriggerSlope.NEGATIVE
    session.configure_trigger_edge(trigger_source = trigger_source, level = trigger_level, trigger_coupling = niscope.TriggerCoupling.LF_REJECT, slope = trigger_slope)
    '''
    print(session.trigger_type)
    print(session.trigger_source)
    print(session.trigger_level)
    print(session.trigger_coupling)
    print(session.trigger_slope)
    print(session.acq_arm_source)
    '''

    from datetime import datetime
    dt0 = datetime.now()
    dt_str = dt0.strftime("%Y%m%d_%H%M%S")[2:]

    for ii in range(nwf):
        print(ii)
        timeout = 100
        with session.initiate():
            #waveforms = session.channels[0].fetch_into(dd0, timeout=5)
            #waveforms = session.channels[1].fetch_into(dd1, timeout=5)
            #waveforms = session.channels[0].fetch_into(dd, timeout=5, num_records=num_records)

            #ax.plot(np.arange(npts), dd, 'o')

            waveforms = session.channels[0,1].fetch(timeout=timeout)
            dt_this = datetime.now()
        ch0.append(waveforms[0].samples[:npts])
        ch1.append(waveforms[1].samples[:npts])
        deltat.append(dt_this - dt0)

dt1 = datetime.now()
daq_rate = nwf/(dt1.timestamp() - dt0.timestamp())
print(f', rate: {daq_rate:.2f} Hz')

ch0 = np.array(ch0, dtype = np.float32)
ch1 = np.array(ch1, dtype = np.float32)
deltat = np.array(deltat)
#print(ch0.shape)

np.savez(f'wf_{dt_str}_{daq_rate:.2f}Hz.npz', ch0 = ch0, ch1 = ch1, npts = npts, sample_rate = sample_rate, ref_position = ref_position, daq_rate = daq_rate, deltat = deltat)
#ax.plot(np.arange(npts), dd, 'o')
#plt.show()
