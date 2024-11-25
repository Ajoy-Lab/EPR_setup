# I am using only external trigger for DAC
# I am using only task trigger for ADC
# use DUC for coherent pulse
# no dc_offset is used
# use 8x interpolation, the IQ rate 1GS/s, DAC rate 8GS/s
# in one segment the IQ complex point should be a multiple of 16, and the total amount should be more than 32
# digital channel 1 can only be task trigger by output ch1, ch2, and digital channel 2 can only be task trigger by output ch3, ch4,
#  In IQ mode, the output V_{pp} has a 3dB reduction
# =============================================================================
from typing import Optional
from astropy.units.quantity_helper.helpers import degree_to_radian_ufuncs
from fontTools.misc.cython import returns
from AWG_function.teproteus import TEProteusAdmin as TepAdmin
import numpy as np

# =============================================================================
# constant, do not change if you are unsure!!!!
# =============================================================================
SampleRateDAC_IQ = 1E9
SampleRateDAC = SampleRateDAC_IQ * 8  # int x8

DAC_granularity = 16  # 16 complex points
DAC_min_segment_length = 32

DAC_min_segment_length_ns = DAC_min_segment_length / (SampleRateDAC_IQ / 1E9)
DAC_granularity_ns = DAC_granularity / (SampleRateDAC_IQ / 1E9)

SampleRateADC = SampleRateDAC / 4
SampleRateADC_IQ = SampleRateADC / 16  # decimation x16

ADC_granularity = 48  # 48 complex points
ADC_granularity_ns = ADC_granularity / (SampleRateADC_IQ / 1E9)

digitizer_system_delay = 560E-9


# =============================================================================
# general functions
# =============================================================================
def analog_to_digital_IQ(I_analog_normalized: np.ndarray,
                         Q_analog_normalized: np.ndarray,
                         DAC_res: Optional[int] = 16
                         ):
    '''
    do analog to digital conversion before streaming waveform data to Proteus
    convert to IQ form

    :param I_analog_normalized: the I component analog waveform, normalized
    :param Q_analog_normalized: the Q component analog waveform, normalized
    :param DAC_res: the resolution of DAC, for P9484 is it 16 bits
    :return: the converted digital waveform
    '''
    dac_resolution = 2 ** DAC_res - 1
    data_type = np.uint16
    V_min = -1 / 2
    I_digital = (I_analog_normalized - V_min) * (dac_resolution - 1)
    Q_digital = (Q_analog_normalized - V_min) * (dac_resolution - 1)
    I_digital = I_digital.astype(data_type)
    Q_digital = Q_digital.astype(data_type)
    # convert to IQIQIQIQ...
    arr_tuple = (I_digital, Q_digital)
    IQ_digital = np.vstack(arr_tuple).reshape((-1,), order='F')
    return IQ_digital


def check_error_SCPI(inst,
                     Error_message: str
                     ):
    '''
    check error, if yes, stop
    :param inst:
    :return: None
    '''
    Error_message_SCPI = inst.send_scpi_query(':SYST:ERR?')
    assert Error_message_SCPI == '0, no error', Error_message + '\n' + Error_message_SCPI
    return


# =============================================================================
# instrument control
# =============================================================================
def connect_PXI(sid: Optional[int] = 8
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
    check_error_SCPI(inst, 'unable to connect to Proteus')
    return inst


def configurate_all_DAC(inst,
                        ext_trigger_source_DAC: np.ndarray[int],
                        carrier_frequency: np.ndarray[float],
                        output_Vpp: Optional[np.ndarray[np.float64]] = np.array([0.2, 0.2, 0.2, 0.2]),
                        ):
    '''
    configure the DAC output, set up IQ mode, DUC mode, sampling frequency, output level

    :param inst: the inst object
    :param ext_trigger_source_DAC: the trigger source for each channel: 0:not configurate, 1: TRG1, 2: TRG2
    :param carrier_frequency: the carrier frequencies for up conversion, [ch1,ch2,ch3,ch4], unit Hz
    :param output_Vpp: the output level of after IQ, [ch1,ch2,ch3,ch4], unit V
    :return: None
    '''
    assert (len(ext_trigger_source_DAC) == 4) & (
        np.all(np.isin(ext_trigger_source_DAC, [0, 1, 2]))), "ext trigger source has to be TRG1/TRG2"
    print('DAC sample rate {0}GS/s, SCLK {1}GS/s'.format(SampleRateDAC_IQ / 1E9, SampleRateDAC / 1E9))
    print('DAC minimun segment length {0}ns, segment granuality {1}ns' \
          .format(DAC_min_segment_length_ns, DAC_granularity_ns))
    output_Vpp_IQ = output_Vpp * 2  # the waveform after IQ modulation have 3dB reduction

    for ch in np.arange(4) + 1:
        assert (output_Vpp_IQ[ch - 1] <= 1.2) & (
                output_Vpp_IQ[ch - 1] >= 1E-3), "Vpp value not allowed, have to be between 0.5E-3 and 0.6"
        assert (carrier_frequency[ch - 1] <= 1E9) & (
                carrier_frequency[ch - 1] >= 0), "NCO freq should be between DC to 1GHz"
        inst.send_scpi_cmd(':INST:CHAN {0}'.format(ch))
        inst.send_scpi_cmd(':FREQ:RAST {0}'.format(2.5E9))  # force DAC to 16 bits
        inst.send_scpi_cmd(':INIT:CONT OFF')
        inst.send_scpi_cmd(':TRAC:DEL:ALL')
        if ext_trigger_source_DAC[ch - 1] != 0:
            inst.send_scpi_cmd(':TRIG:SOUR:ENAB TRG{}'.format(ext_trigger_source_DAC[ch - 1]))
        # configurate IQ mode
        inst.send_scpi_cmd(':SOUR:MODE DUC')
        inst.send_scpi_cmd(':SOUR:INT X8')
        inst.send_scpi_cmd(':SOUR:IQM ONE')
        inst.send_scpi_cmd(':FREQ:RAST {0}'.format(SampleRateDAC))
        inst.send_scpi_cmd(':SOUR:NCO:CFR1 {0}'.format(carrier_frequency[ch - 1]))
        inst.send_scpi_cmd(':VOLT {}'.format(output_Vpp_IQ[ch - 1]))

    check_error_SCPI(inst, 'DAC configuration error')
    print('DAC configuration successful')
    return


def configurate_all_trigger(inst,
                            trigger_level: Optional[np.ndarray[np.float64]] = np.array([0.5, 0.5]),
                            trigger_delay: Optional[np.ndarray[np.float64]] = np.array([0E-9, 0E-9]),
                            ):
    '''
    configure the trigger

    :param inst: the inst object
    :param trigger_level: voltage needed to trigger
    :param trigger_delay: delay between receiving trigger and execute
    :return: None
    '''
    for trigger_ch in np.arange(2) + 1:
        inst.send_scpi_cmd(':TRIG:SELECT TRG{}'.format(trigger_ch))
        inst.send_scpi_cmd(':TRIG:LTJ ON')
        inst.send_scpi_cmd(':TRIG:LEV {}'.format(trigger_level[trigger_ch - 1]))
        inst.send_scpi_cmd(':TRIG:DEL {}'.format(trigger_delay[trigger_ch - 1]))
        inst.send_scpi_cmd(':TRIG:SLOP POS')
        inst.send_scpi_cmd(':TRIG:COUP ON')
        inst.send_scpi_cmd(':TRIG:STAT ON')
    check_error_SCPI(inst, 'trigger configuration error')
    print('trigger configuration successful')
    return


class segment(object):
    ''' the building block of pulse sequence '''

    def __init__(self,
                 address: dict
                 ):
        '''
        :param address: key={'inst','channel','segment'}
        '''
        self.address = address
        self.I = np.array([])
        self.Q = np.array([])
        return

    def default_pulse(self,
                      length: int,
                      amplitude: np.float64,
                      phase: np.float64,
                      type: str):
        '''
        this function defines a simple pulse (empty, sine, gaussian etc. ).
        The defined pulse is add to the property I and Q
        here assuming that RF=I*cos-Q*sin

        :param length: the length of the pulse, in unit of 1ns
        :param amplitude: normalized amplitude (from -0.5 to +0.5)
        :param phase: the phase of the pulse, in degree
        :param type: empty: a delay, sine: a sine wave, other options are under construction...
        :return: None
        '''
        assert np.abs(amplitude) <= 0.5, 'amplitude must be between -0.5 and +0.5'
        if type == 'empty':
            I = np.zeros(length)
            Q = np.zeros(length)
        elif type == 'sine':
            degree_to_radian = np.pi / 180
            I = amplitude * np.cos(degree_to_radian * phase)
            Q = amplitude * np.sin(degree_to_radian * phase)
        else:
            raise ValueError("are you entering the correct pulse type?")
        self.I = np.append(self.I, I)
        self.Q = np.append(self.Q, Q)
        return

    def custom_pulse(self,
                     I: np.ndarray[np.float64],
                     Q: np.ndarray[np.float64]
                     ):
        '''
        customize a pulse. Need to input I, Q component as np array

        :param I: I component, in analog form, normalized
        :param Q: Q component, in analog form, normalized
        :return: None
        '''
        assert I.shape == Q.shape, 'I,Q components must have same length'
        assert (np.max(np.abs(I)) <= 1) & (np.max(np.abs(Q)) <= 1), 'should input normalized I, Q components'
        self.I = np.append(self.I, I)
        self.Q = np.append(self.Q, Q)
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
        assert (np.max(np.abs(self.I)) <= 1 / 2) & (
                np.max(np.abs(self.Q)) <= 1 / 2), 'should input normalized I, Q components'

        if len(self.I) < DAC_min_segment_length:
            filled_number = DAC_min_segment_length - len(self.I)
            fill_array = np.zeros(filled_number, dtype=np.float64)
            self.I = np.append(self.I, fill_array)
            self.Q = np.append(self.Q, fill_array)
            print('Warning: Pulse length shorter than {0}, has zero filled {1} numbers'.format(DAC_min_segment_length,
                                                                                               filled_number))

        remainder = len(self.I) % DAC_granularity
        if remainder != 0:
            filled_number = DAC_granularity - remainder
            fill_array = np.zeros(filled_number, dtype=np.float64)
            self.I = np.append(self.I, fill_array)
            self.Q = np.append(self.Q, fill_array)
            print('Warning: Pulse length not a multiple of {0}, has zero filled {1} numbers'.format(DAC_granularity,
                                                                                                    filled_number))

        # analog to digital conversion
        inst = self.address['inst']
        ch = self.address['channel']
        segnum = self.address['segment']
        IQ_digital = analog_to_digital_IQ(self.I, self.Q)

        # download waveform to FPGA
        inst.send_scpi_cmd(':INST:CHAN {0}'.format(ch))
        inst.send_scpi_cmd(':TRAC:DEF {0}, {1}'.format(segnum, len(IQ_digital)))
        inst.send_scpi_cmd(':TRAC:SEL {0}'.format(segnum))
        # Increase the timeout before writing binary-data:
        inst.timeout = 30000
        # Send the binary-data with *OPC? added to the beginning of its prefix.
        inst.write_binary_data('*OPC?; :TRAC:DATA', IQ_digital)
        # Set normal timeout
        inst.timeout = 10000
        check_error_SCPI(inst, 'IQ waveform streaming error, channel:{0}, segment:{1}'.format(ch, segnum))
        print('channel {0}, segment {1} downloaded'.format(ch, segnum))
        return


class task_table(object):
    '''task table put segment in specific order to execute'''

    def __init__(self,
                 address: dict,
                 length: int
                 ):
        '''
        reset the task table at a specific channel and define a new one

        :param address: key={'inst','channel'}
        :param length: length of the task table
        :return: None
        '''
        self.address = address
        self.length = length
        # initialize the channel task table
        inst = self.address['inst']
        ch = self.address['channel']
        inst.send_scpi_cmd(':INST:CHAN {0}'.format(ch))
        inst.send_scpi_cmd(':TASK:ZERO:ALL')
        inst.send_scpi_cmd(':TASK:COMP:LENG {}'.format(self.length))
        check_error_SCPI(inst, 'Defining task table at channel {} failed'.format(ch))
        return

    def new_task(self,
                 tasknum: int,
                 segnum: int,
                 ext_trigger_ch: int,
                 digitizer_trigger: bool,
                 next_task: int,
                 loop: Optional[int] = 1,
                 delay: Optional[np.float64] = 0,
                 ):
        '''
        define a new task

        :param tasknum: the task number assigned to it
        :param segnum: the segment assigned to this task
        :param ext_trigger_ch: the external trigger channel that trigger this channel. If not trigger, set to 0
        :param digitizer_trigger: whether trigger digitizer or not
        :param next_task: the next task to execute
        :param loop: how many loops for this task
        :param delay: delay time before executing next task, in unit of ns. resolution 125ps,
                        maximum length 8192ns.

        notice
        :return: None
        '''

        inst = self.address['inst']
        ch = self.address['channel']
        inst.send_scpi_cmd(':INST:CHAN {0}'.format(ch))
        assert (tasknum >= 1) & (tasknum <= self.length), 'input task number out of range, increase task table length'
        inst.send_scpi_cmd(':TASK:COMP:SEL {0}'.format(tasknum))
        inst.send_scpi_cmd(':TASK:COMP:SEGM {0}'.format(segnum))
        if ext_trigger_ch != 0:
            inst.send_scpi_cmd(':TASK:COMP:ENAB TRG{}'.format(ext_trigger_ch))
        if digitizer_trigger:
            inst.send_scpi_cmd(':TASK:COMP:DTR ON')
        inst.send_scpi_cmd(':TASK:COMP:NEXT1 {0}'.format(next_task))
        inst.send_scpi_cmd(':TASK:COMP:LOOP {0}'.format(loop))

        # define delay before next task
        delay_in_SCLK = round(delay * 8)
        assert (delay_in_SCLK >= 0) & (delay_in_SCLK <= 65536), \
            'delay should be between 0ns and 8192ns, use external trigger if larger delay needed'
        inst.send_scpi_cmd(':TASK:COMP:DEL {0}'.format(delay_in_SCLK))
        check_error_SCPI(inst, 'Defining new task in channel {0}, task number {1} failed'.format(ch, tasknum))
        return

    def download_task_table(self):
        '''write the task table to Proteus after defining them'''
        inst = self.address['inst']
        ch = self.address['channel']
        inst.send_scpi_cmd(':INST:CHAN {0}'.format(ch))
        inst.send_scpi_cmd(':TASK:COMP:WRITE')
        inst.send_scpi_cmd(':SOUR:FUNC:MODE TASK')
        inst.send_scpi_cmd(':OUTP ON')
        check_error_SCPI(inst, 'download task table at channel {0} failed'.format(ch))
        print('channel {} task table downloaded'.format(ch))
        return


class digitizer(object):
    '''one fo the two digitizers channel'''

    def __init__(self,
                 address: dict,
                 task_trigger_channel: int,
                 carrier_frequency: np.ndarray[float],
                 numframes: int,
                 framelen: int,
                 delay: Optional[np.float64] = digitizer_system_delay
                 ):
        '''
        configurate a specific digitizer, and start acquisition

        :param address: the address of the digitizer, key={'inst', 'channel'}
        :param task_trigger_channel: the trigger DAC channel of the digitizer
        :param carrier_frequency: the carrier frequency for DDC
        :param numframes: the number of frames to acquire
        :param framelen: the length of each frame
        :delay: the delay time before acquisition. This it usually set to the system delay of this digitizer
        :return: None
        '''
        print('digitizer sample rate {0}GS/s, SCLK {1}GS/s'.format(SampleRateADC / 1E9, SampleRateADC_IQ / 1E9))
        print('digitizer granularity {}ns'.format(ADC_granularity_ns))

        remainder = framelen % ADC_granularity
        if remainder != 0:
            filled_number = ADC_granularity - remainder
            framelen += filled_number
            print('Warning, frame length not a multiple of {0}, has filled {1} numbers'.format(ADC_granularity,
                                                                                               filled_number))

        self.address = address
        self.task_trigger_channel = task_trigger_channel
        self.carrier_frequency = carrier_frequency
        self.numframes = numframes
        self.framelen = framelen
        self.SampleRateADC = SampleRateADC
        self.delay = delay

        inst = address['inst']
        ch = address['channel']
        assert (task_trigger_channel >= 2 * ch - 1) & (task_trigger_channel <= 2 * ch) \
            , 'digitizer channel {0} can only be triggered by DAC channel {1}, {2}'.format(ch, 2 * ch - 1, 2 * ch)

        totlen = numframes * framelen
        stored_waveform = np.zeros(totlen, dtype=np.uint32)
        self.stored_waveform = stored_waveform
        print('Channel {0} aquisition frame Length {1}, frame number {2}' \
              .format(ch, framelen / 2 / SampleRateADC * 16, numframes))  # x16 decimation

        # overall digitizer settings
        inst.send_scpi_cmd(':DIG:MODE DUAL')
        inst.send_scpi_cmd(':DIG:FREQ  {0}'.format(SampleRateADC))

        # setting regarding the specific channel
        inst.send_scpi_cmd(':DIG:CHAN:SEL {}'.format(ch))
        inst.send_scpi_cmd(':DIG:INIT OFF')
        inst.send_scpi_cmd(':DIG:DDC:MODE COMP')
        inst.send_scpi_cmd(':DIG:DDC:CFR1 {0}'.format(carrier_frequency))
        inst.send_scpi_cmd(':DIG:DDC:PHAS1 0')
        inst.send_scpi_cmd(':DIG:DDC:CLKS AWG')
        inst.send_scpi_cmd(':DIG:CHAN:STATE ENAB')

        # delay is for individual task list, but not for individual digitizer
        inst.send_scpi_cmd(':DIG:TRIG:SOURCE TASK{}'.format(task_trigger_channel))
        inst.send_scpi_cmd(':INST:CHAN:SEL {}'.format(task_trigger_channel))
        inst.send_scpi_cmd(':DIG:TRIG:AWG:TDEL {0}'.format(delay))
        inst.send_scpi_cmd(':DIG:ACQuire:FRAM:DEF {0},{1}'.format(numframes, framelen))

        # Select the frames for the capturing
        capture_first, capture_count = 1, numframes
        inst.send_scpi_cmd(':DIG:ACQuire:FRAM:CAPT {0},{1}'.format(capture_first, capture_count))

        # trigger the digitizer
        inst.send_scpi_cmd(':DIG:INIT ON')
        inst.send_scpi_cmd('*TRG')

        check_error_SCPI(inst, 'digitizer configuration falied')
        return

    def inquire_acquisition(self):
        '''
        inquire if acquisition has been done, and the frames done

        :return: if acquisition has been done, finish_flag=1, otherwise 0. Also return the frames acquired
        '''
        inst = self.address['inst']
        ch = self.address['channel']
        inst.send_scpi_cmd(':DIG:CHAN:SEL {}'.format(ch))
        resp = inst.send_scpi_query(':DIG:ACQuire:FRAM:STATus?')
        framesParam = resp.split(",")
        finish_flag = int(framesParam[1])
        frames_acquired = int(framesParam[3])
        return finish_flag, frames_acquired

    def read_digitizer_data(self,
                            ):
        '''
        stop the digitizer, and read data from it

        :return: None
        '''
        inst = self.address['inst']
        ch = self.address['channel']
        inst.send_scpi_cmd(':DIG:CHAN:SEL {}'.format(ch))
        inst.send_scpi_cmd(':DIG:INIT OFF')
        inst.send_scpi_cmd(':DIG:DATA:SEL ALL')
        inst.send_scpi_cmd(':DIG:DATA:TYPE FRAM')

        # Get the total data size (in bytes)
        resp = inst.send_scpi_query(':DIG:DATA:SIZE?')
        num_bytes = np.uint32(resp)
        print('Total read size in bytes: ' + resp)
        # in complex mode, 1 IQ point takes 4 Bytes

        inst.send_scpi_cmd(':DIG:CHAN:SEL {}'.format(ch))
        wavlen = num_bytes // 2
        rc = inst.read_binary_data(':DIG:DATA:READ?', self.stored_waveform, wavlen)
        inst.send_scpi_cmd(':DIG:ACQ:ZERO:ALL 0')

        self.stored_waveform = np.int32(self.stored_waveform) - 16384

        self.wavI = self.stored_waveform[0::2]
        self.wavQ = self.stored_waveform[1::2]
        self.wavI = self.wavI.astype(float)
        self.wavQ = self.wavQ.astype(float)
        check_error_SCPI(inst, 'reading digitizer data failed')
        print('acquired {} complex points'.format(len(self.wavI)))
        return
