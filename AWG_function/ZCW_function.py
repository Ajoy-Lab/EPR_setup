# I am using only external trigger for DAC
# I am using only task trigger for ADC
# use DUC for coherent pulse
# no dc_offset is used
# use 8x interpolation, the IQ rate 1GS/s, DAC rate 8GS/s
# in one segment the IQ complex point should be a multiple of 16, and the total amount should be more than 32
# digital channel 1 can only be task trigger by output ch1, ch2, and digital channel 2 can only be task trigger by output ch3, ch4,
#  In IQ mode, the output V_{pp} has a 3dB reduction



from typing import Optional

from astropy.units.quantity_helper.helpers import degree_to_radian_ufuncs
from fontTools.misc.cython import returns

from AWG_function.teproteus import TEProteusAdmin as TepAdmin
import numpy as np

# constant, do not change if you are unsure!!!!
SampleRateIQ=1E9
granularity=16
min_segment_length=32

# global variable

class segment(object):
    ''' the building block of pulse sequence '''
    def __init__(self,
                 address:dict
                 ):
        '''
        :param address: key={'inst','channel','segment'}
        '''
        self.address = address
        self.I = np.array([])
        self.Q = np.array([])
        return

    def default_pulse(self,
                      length:np.int64,
                      amplitude:np.float64,
                      phase:np.float64,
                      type:str):
        '''
        this function defines a simple pulse (empty, sine, gaussian etc. ).
        The defined pulse is add to the property I and Q
        here assuming that RF=I*cos-Q*sin

        :param length: the length of the pulse, in unit of 1ns
        :param amplitude: normalized amplitude, when set to 1, output voltage Vpp=channel Vpp
        :param phase: the phase of the pulse, in degree
        :param type: empty: a delay, sine: a sine wave, other options are under construction...
        :return: None
        '''
        if type=='empty':
            I=np.zeros(length)
            Q=np.zeros(length)
        elif type=='sine':
            degree_to_radian=np.pi/180
            I=amplitude*np.cos(degree_to_radian*phase)
            Q=amplitude*np.sin(degree_to_radian*phase)
        else:
            raise ValueError("are you entering the correct pulse type?")
        self.I = np.append(self.I,I)
        self.Q = np.append(self.Q,Q)
        return

    def custom_pulse(self,
                     I:np.array[np.float64],
                     Q:np.array[np.float64]
                     ):
        '''
        customize a pulse. Need to input I, Q component as np array

        :param I: I component, in analog form, normalized
        :param Q: Q component, in analog form, normalized
        :return: None
        '''
        assert I.shape == Q.shape, 'I,Q components must have same length'
        assert (np.max(np.abs(I))<=1)&(np.max(np.abs(Q))<=1), 'should input normalized I, Q components'
        self.I = np.append(self.I,I)
        self.Q = np.append(self.Q,Q)
        return

    def compile_and_download_segment(self):
        '''
        convert the pulses from analog to digital format.
        Check if the length of the pulses meet the requirements. If not, zero fill.
        Download the waveform to Proteus

        :return: None
        '''

        # checking errors
        assert self.I.shape == self.Q.shape, 'I,Q components must have same length'
        assert (np.max(np.abs(self.I)) <= 1) & (np.max(np.abs(self.Q)) <= 1), 'should input normalized I, Q components'

        if len(self.I) < min_segment_length:
            filled_I = np.zeros(min_segment_length,dtype=np.float64)
            filled_Q = np.zeros(min_segment_length, dtype=np.float64)
            filled_I[:len(self.I)] = self.I
            filled_Q[:len(self.Q)] = self.Q
            self.I = filled_I
            self.Q = filled_Q
            print('Warning: Pulse length shorter than {}, has zero filled'.format(min_segment_length))

        remainder=len(self.I) % granularity
        if remainder!=0:
            fill_array=np.zeros(remainder,dtype=np.float64)
            self.I = np.append(self.I,fill_array)
            self.Q = np.append(self.Q,fill_array)
            print('Warning: Pulse length not a multiple of {0}, has zero filled {1} numbers'.format(granularity,remainder))

        # analog to digital conversion

        inst=self.address['inst']
        ch=self.address['channel']
        segment=self.address['segment']

        inst.send_scpi_cmd(':INST:CHAN {0}'.format(ch))


131111




def connect_PXI(sid:Optional[int]=8
                ):
    '''
    This function connects to the instrument. And reset.

    :param sid: the PCIE port connecting to computer, on EPR computer it is 8
    :return: the inst object
    '''
    # connect and initialize
    admin = TepAdmin()  # required to control PXI module
    inst = admin.open_instrument(slot_id=sid)
    resp = inst.send_scpi_query("*IDN?")
    print('connected to: ' + resp)  # Print *IDN
    inst.send_scpi_cmd('*CLS; *RST')
    check_error_SCPI(inst,'unable to connect to Proteus')
    return inst

def configurate_all_DAC(inst,
                    ext_trigger_source_DAC: np.ndarray[np.int64],
                    carrier_frequency:np.ndarray[float],
                    output_Vpp:Optional[np.ndarray[np.float64]]=np.array([0.2,0.2,0.2,0.2]),
                    SampleRateIQ:Optional[float]=SampleRateIQ
                    ):
    '''
    configure the DAC output, set up IQ mode, DUC mode, sampling frequency, output level

    :param inst: the inst object
    :param ext_trigger_source_DAC: the trigger source for each channel: 0:not configurate, 1: ch1, 2: ch2
    :param carrier_frequency: the carrier frequencies for up conversion, [ch1,ch2,ch3,ch4], unit Hz
    :param output_Vpp: the output level of after IQ, [ch1,ch2,ch3,ch4], unit V
    :param SampleRateIQ: the DAC sample rate before interpolation
    :return: None
    '''
    assert (len(ext_trigger_source_DAC)==4)& (
        np.all(np.isin(ext_trigger_source_DAC, [0, 1, 2]))), "ext trigger source has to be TRG1/TRG2"
    SampleRateDAC=SampleRateIQ*8 # X8 interpolation
    output_Vpp_IQ=output_Vpp*2 # the waveform after IQ modulation have 3dB reduction


    for ch in np.arange(4)+1:
        assert (output_Vpp_IQ[ch-1] <= 1.2) & (
                    output_Vpp_IQ[ch-1] >= 1E-3), "Vpp value not allowed, have to be between 0.5E-3 and 0.6"
        assert (carrier_frequency[ch-1] <= 1E9) & (
                    carrier_frequency[ch-1] >= 0), "NCO freq should be between DC to 1GHz"
        inst.send_scpi_cmd(':INST:CHAN {0}'.format(ch))
        inst.send_scpi_cmd(':FREQ:RAST {0}'.format(2.5E9))  # force DAC to 16 bits
        inst.send_scpi_cmd(':INIT CONT OFF')
        inst.send_scpi_cmd(':TRAC:DEL:ALL')
        if ext_trigger_source_DAC[ch-1]!=0:
            inst.send_scpi_cmd(':TRIG:SOUR:ENAB TRG{}'.format(ext_trigger_source_DAC[ch-1]))
        # configurate IQ mode
        inst.send_scpi_cmd(':SOUR:MODE DUC')
        inst.send_scpi_cmd(':SOUR:INT X8')
        inst.send_scpi_cmd(':SOUR:IQM ONE')
        inst.send_scpi_cmd(':FREQ:RAST {0}'.format(SampleRateDAC))
        inst.send_scpi_cmd(':SOUR:NCO:CFR1 {0}'.format(carrier_frequency[ch-1]))
        inst.send_scpi_cmd(':VOLT {}'.format(output_Vpp_IQ[ch-1]))
        inst.send_scpi_cmd(':OUTP ON')
    check_error_SCPI(inst,'DAC configuration error')
    return

def configurate_all_trigger(inst,
                        trigger_level:Optional[np.ndarray[np.float64]]=np.array([2.0,2.0]),
                        trigger_delay:Optional[np.ndarray[np.float64]]=np.array([0E-9,0E-9]),
                        ):
    '''
    configure the trigger

    :param inst: the inst object
    :param trigger_level: voltage needed to trigger
    :param trigger_delay: delay between receiving trigger and execute
    :return: None
    '''
    for trigger_ch in np.arange(2)+1:
        inst.send_scpi_cmd(':TRIG:SELECT TRG{}'.format(trigger_ch))
        inst.send_scpi_cmd(':TRIG:LTJ ON')
        inst.send_scpi_cmd(':TRIG:LEV {}'.format(trigger_level[trigger_ch-1]))
        inst.send_scpi_cmd(':TRIG:DEL {}'.format(trigger_delay[trigger_ch-1]))
        inst.send_scpi_cmd(':TRIG:SLOP POS')
        inst.send_scpi_cmd(':TRIG:COUP ON')
        inst.send_scpi_cmd(':TRIG:STAT ON')
    check_error_SCPI(inst, 'trigger configuration error')
    return

def create_waveform():
    return

def download_all_segment(address:dict,
                   phase:float,
                   length:float,
                   pulse_mod: Optional[str]='No',
                   amplitude: Optional[np.float64]=1,
                   ):
    '''
    create a segment in the Proteus. Remember that CH1/CH2 CH3/CH4 share the same segment memory

    :param address: key={'inst', 'channel_no', 'segment_no'}
    :param phase: phase of the pulse in degrees
    :param length: length of pulse in
    :param pulse_mod: for shaped pulse, under construction...
    :param amplitude: normalized amplitude of the pulse, volt=amplitude * volt of the DAC channel
    :return: None
    '''

    time=np.linspace(0,length,endpoint=False)
    if pulse_mod=='sine':
        wave_analog_normalized=np.sin(2*np.pi*freq)


    return

def analog_to_digital(wave_analog_normalized:np.ndarray,
                      DAC_res:Optional[np.int8]=16
                      ):
    '''
    do analog to digital conversion before streaming waveform data to Proteus

    :param wave_analog_normalized: the analog waveform, normalized
    :param DAC_res:
    :return:
    '''
    dac_resolution=2**DAC_res
    V_min=-1/2
    wave_digital=(wave_analog_normalized-V_min)/1*(dac_resolution-1)
    return wave_digital




def build_task_table():
    return 0

def check_error_SCPI(inst,
                Error_message:str
                ):
    '''
    check error, if yes, stop
    :param inst:
    :return: None
    '''
    Error_message_SCPI=inst.send_scpi_query(':SYST:ERR?')
    assert Error_message_SCPI=='0, no error', Error_message+'\n'+Error_message_SCPI
    return

def digital_to_analog_conventional(wave_digital:np.ndarray[float],
                                   ):
    max