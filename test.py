from AWG_function.ZCW_function import *

inst=connect_PXI()
configurate_DAC(inst,[1E6,1E6,1E6,1E6])