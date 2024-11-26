# =============================================================================
# some default setting:
# only task mode
#
# only ext trigger for DAC
# only task trigger for ADC
#
# using DUC and DDC, synchronized for coherent control and readout
#
# for DAC, use ONE mode, use 8x interpolation, IQ rate 1GS/s, DAC rate 8GS/s
# for ADC, use DUAL mode, 16x decimation, IQ rate 125MS/s, ADC rate 2GS/s
#
# for DAC segment, the IQ complex point should be a multiple of 16, and the total amount should be more than 32
# for ADC frame, the IQ complex point should be a multiple of 48
#
# always finish programming one channel, and then go to another
# if one coherent DDC and DUC, they should have the same NCO freq. And DAC clock should be a multiple of ADC clock

# =============================================================================
# common:
# SCLK for every DAC channel, IQ mode and interpolation factor
# SCLK for every ADC channel, IQ mode and decimation factor
# triggering level for every trigger

# semi-common
# digitizer 1 can be task triggered by DAC 1, 2, digitizer 2 can be task trigger by DAC 3, 4

# different
# all the trigger can have different triggering level
# all the DAC channel have independent NCO
# all the ADC channel have independent NCO

# =============================================================================
from typing import Optional
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

decimation_ADC = 16
SampleRateADC = SampleRateDAC / 4
SampleRateADC_IQ = SampleRateADC / decimation_ADC

ADC_granularity = 48  # 48 complex points
ADC_granularity_ns = ADC_granularity / (SampleRateADC_IQ / 1E9)

digitizer_system_delay = 0


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


def configurate_one_DAC(inst,
                        channel: int,
                        carrier_frequency: float,
                        trigger_channel: int,
                        trigger_setting: Optional[dict] = {'trigger_level': 0.5, 'trigger_delay': 0},
                        output_Vpp: Optional[float] = 0.5,
                        ):
    '''
    configure the DAC output, set up IQ mode, DUC mode, sampling frequency, output level.
    Note: always finish configuring one channel first, and then another

    :param inst: the inst object
    :param channel: the DAC channel number
    :param carrier_frequency: the carrier frequencies for up conversion, unit Hz
    :param trigger_channel: the trigger channel number
    :param trigger_setting: key={'trigger_level', 'trigger_delay'}
    :param output_Vpp: the output level of after IQ unit V
    :return: None
    '''
    assert (trigger_channel == 1) | (trigger_channel == 1), "ext trigger source has to be TRG1/TRG2"
    print('DAC sample rate {0:.2f}GS/s, SCLK {1:.2f}GS/s'.format(SampleRateDAC_IQ / 1E9, SampleRateDAC / 1E9))
    print('DAC minimum segment length {0:.1f}ns, segment granularity {1:.1f}ns' \
          .format(DAC_min_segment_length_ns, DAC_granularity_ns))

    assert (output_Vpp <= 0.5) & (
            output_Vpp >= 1E-3), "Vpp value not allowed, have to be between 0.5E-3 and 0.25"
    assert (carrier_frequency <= 1E9) & (
            carrier_frequency >= 0), "NCO freq should be between DC to 1GHz"

    # initialize DAC
    inst.send_scpi_cmd(':INST:CHAN {0}'.format(channel))
    inst.send_scpi_cmd(':FREQ:RAST {0}'.format(2.5E9))  # force DAC to 16 bits
    inst.send_scpi_cmd(':INIT:CONT ON')
    inst.send_scpi_cmd(':TRAC:DEL:ALL')
    inst.send_scpi_cmd(':TRIG:SOUR:ENAB TRG{}'.format(trigger_channel))

    check_error_SCPI(inst, 'channel {} DAC initialization error'.format(channel))

    # configurate trigger
    inst.send_scpi_cmd(':TRIG:SELECT TRG{}'.format(trigger_channel))
    inst.send_scpi_cmd(':TRIG:LTJ ON')
    inst.send_scpi_cmd(':TRIG:LEV {}'.format(trigger_setting['trigger_level']))
    inst.send_scpi_cmd(':TRIG:DEL {}'.format(trigger_setting['trigger_delay']))
    inst.send_scpi_cmd(':TRIG:SLOP POS')
    inst.send_scpi_cmd(':TRIG:COUP ON')
    inst.send_scpi_cmd(':TRIG:STAT ON')

    check_error_SCPI(inst, 'channel {} trigger configuration error'.format(channel))

    # configurate IQ mode
    inst.send_scpi_cmd(':SOUR:MODE DUC')
    inst.send_scpi_cmd(':SOUR:INT X8')
    inst.send_scpi_cmd(':SOUR:IQM ONE')
    inst.send_scpi_cmd(':FREQ:RAST {0}'.format(SampleRateDAC))
    inst.send_scpi_cmd(':SOUR:NCO:CFR1 {0}'.format(carrier_frequency))
    inst.send_scpi_cmd(':NCO:SIXD1 ON')
    inst.send_scpi_cmd(':VOLT {}'.format(output_Vpp))

    check_error_SCPI(inst, 'channel {} IQ configuration error'.format(channel))
    print('DAC configuration successful')
    return


class segment(object):
    ''' the building block of pulse sequence '''

    def __init__(self,
                 inst,
                 segnum: int
                 ):
        '''
        :param inst: the inst object
        :param segnum: the segment number being defined
        '''
        self.inst = inst
        self.segnum = segnum
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
            I = amplitude * np.cos(degree_to_radian * phase) * np.ones(length)
            Q = amplitude * np.sin(degree_to_radian * phase) * np.ones(length)
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

        :param I: I component, in analog form, normalized to -0.5 to +0.5
        :param Q: Q component, in analog form, normalized
        :return: None
        '''
        assert I.shape == Q.shape, 'I,Q components must have same length'
        assert (np.max(np.abs(I)) <= 0.5) & (np.max(np.abs(Q)) <= 0.5), 'should input normalized I, Q components'
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

        # analog to digital conversion
        inst = self.inst
        segnum = self.segnum
        IQ_digital = analog_to_digital_IQ(self.I, self.Q)

        # safety check
        assert self.I.shape == self.Q.shape, 'I,Q components must have same length'
        assert (np.max(np.abs(self.I)) <= 1 / 2) & (
                np.max(np.abs(self.Q)) <= 1 / 2), 'should input normalized I, Q components'

        signal_amplitude = np.sqrt((2 * self.I) ** 2 + (2 * self.Q) ** 2)
        volt = float(inst.send_scpi_query(':VOLT?'))
        assert np.max(
            signal_amplitude * volt) <= 0.5, 'I, Q component not orthogonal, may be clipping happening, tune down voltage'

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

        # download waveform to FPGA
        inst.send_scpi_cmd(':TRAC:DEF {0}, {1}'.format(segnum, len(IQ_digital)))
        inst.send_scpi_cmd(':TRAC:SEL {0}'.format(segnum))
        # Increase the timeout before writing binary-data:
        inst.timeout = 30000
        # Send the binary-data with *OPC? added to the beginning of its prefix.
        inst.write_binary_data('*OPC?; :TRAC:DATA', IQ_digital)
        # Set normal timeout
        inst.timeout = 10000
        ch = int(inst.send_scpi_query(':INST:CHAN?'))
        check_error_SCPI(inst, 'IQ waveform streaming error, channel:{0}, segment:{1}'.format(ch, segnum))
        print('channel {0}, segment {1} downloaded, length {2:.1f}ns'.format(ch, segnum,
                                                                             len(self.I) / SampleRateDAC_IQ * 1E9))
        return


class task_table(object):
    '''task table put segment in specific order to execute'''

    def __init__(self,
                 inst,
                 length: int
                 ):
        '''
        reset the task table at a specific channel and define a new one

        :param inst: the inst object
        :param length: length of the task table
        :return: None
        '''
        self.inst = inst
        self.length = length
        # initialize the channel task table
        inst = self.inst
        inst.send_scpi_cmd(':TASK:COMP:LENG {}'.format(self.length))
        ch = int(inst.send_scpi_query(':INST:CHAN?'))
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

        inst = self.inst
        assert (tasknum >= 1) & (tasknum <= self.length), 'input task number out of range, increase task table length'
        inst.send_scpi_cmd(':TASK:COMP:SEL {0}'.format(tasknum))
        inst.send_scpi_cmd(':TASK:COMP:SEGM {0}'.format(segnum))
        if ext_trigger_ch != 0:
            inst.send_scpi_cmd(':TASK:COMP:ENAB TRG{}'.format(ext_trigger_ch))
        if digitizer_trigger:
            inst.send_scpi_cmd(':TASK:COMP:DTR ON')
        inst.send_scpi_cmd(':TASK:COMP:LOOP {0}'.format(loop))
        inst.send_scpi_cmd(':TASK:COMP:NEXT1 {0}'.format(next_task))

        # define delay before next task
        delay_in_SCLK = round(delay * 8)
        assert (delay_in_SCLK >= 0) & (delay_in_SCLK <= 65536), \
            'delay should be between 0ns and 8192ns, use external trigger if larger delay needed'
        inst.send_scpi_cmd(':TASK:COMP:DEL {0}'.format(delay_in_SCLK))
        ch = int(inst.send_scpi_query(':INST:CHAN?'))
        check_error_SCPI(inst, 'Defining new task in channel {0}, task number {1} failed'.format(ch, tasknum))
        return

    def download_task_table(self):
        '''write the task table to Proteus after defining them'''
        inst = self.inst
        inst.send_scpi_cmd(':TASK:COMP:WRITE')
        inst.send_scpi_cmd(':SOUR:FUNC:MODE TASK')
        inst.send_scpi_cmd(':OUTP ON')
        ch = int(inst.send_scpi_query(':INST:CHAN?'))
        check_error_SCPI(inst, 'download task table at channel {0} failed'.format(ch))
        print('channel {} task table downloaded'.format(ch))
        return


class digitizer(object):
    '''one fo the two digitizers channel'''

    def __init__(self,
                 address: dict,
                 task_trigger_channel: int,
                 carrier_frequency: float,
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
        print('digitizer sample rate {0:.2f}GS/s, SCLK {1:.2f}GS/s'.format(SampleRateADC_IQ / 1E9, SampleRateADC / 1E9))
        print('digitizer granularity {:.1f}ns'.format(ADC_granularity_ns))

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
        print('Channel {0} acquisition frame Length {1:.1f} ns, frame number {2}' \
              .format(ch, framelen / 2 / SampleRateADC * decimation_ADC * 1E9, numframes))

        # overall digitizer settings
        inst.send_scpi_cmd(':DIG:MODE DUAL')
        inst.send_scpi_cmd(':DIG:FREQ {0}'.format(SampleRateADC))

        # setting regarding the specific channel
        inst.send_scpi_cmd(':DIG:CHAN:SEL {}'.format(ch))
        inst.send_scpi_cmd(':DIG:DDC:MODE COMP')
        inst.send_scpi_cmd(':DIG:DDC:CFR{0} {1}'.format(ch, carrier_frequency))
        inst.send_scpi_cmd(':DIG:DDC:PHAS1 0')
        inst.send_scpi_cmd(':DIG:DDC:CLKS AWG')
        inst.send_scpi_cmd(':DIG:CHAN:STATE ENAB')

        # delay is for individual task list, but not for individual digitizer
        inst.send_scpi_cmd(':DIG:TRIG:SOURCE TASK{}'.format(task_trigger_channel))
        inst.send_scpi_cmd(':INST:CHAN:SEL {0}'.format(task_trigger_channel))
        inst.send_scpi_cmd(':DIG:TRIG:AWG:TDEL {0}'.format(delay))
        inst.send_scpi_cmd(':DIG:ACQuire:FRAM:DEF {0},{1}'.format(numframes, framelen))

        # Select the frames for the capturing
        capture_first, capture_count = 1, numframes
        inst.send_scpi_cmd(':DIG:ACQuire:FRAM:CAPT {0},{1}'.format(capture_first, capture_count))

        # trigger the digitizer
        inst.send_scpi_cmd(':DIG:INIT OFF')
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
        print('acquired {0} complex points, {1:.1f}ns'.format(len(self.wavI), len(self.wavI) / SampleRateDAC_IQ * 1E9))
        return
