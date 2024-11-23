from AWG_function.teproteus import TEProteusAdmin as TepAdmin

# admin = TepAdmin()  # required to control PXI module
#
# inst = admin.open_instrument(slot_id=8)
# resp = inst.send_scpi_query("*IDN?")
# print('connected to: ' + resp)  # Print *IDN
# inst.send_scpi_cmd('*CLS; *RST')
#
# resp=inst.send_scpi_query(':SYST:ERR?')
#
# print(resp)
#
# inst.send_scpi_cmd(':INST:CHAN {0}'.format(1))
#
# resp=inst.send_scpi_query(':SYST:ERR?')
#
# print(resp)
# inst.send_scpi_cmd(':MODE DUC')
#
# resp=inst.send_scpi_query(':SYST:ERR?')
#
# inst.send_scpi_cmd(':FREQ:RAST 1E9')
# resp=inst.send_scpi_query(':SYST:ERR?')
# print(resp)
# inst.send_scpi_cmd(':SOUR:INT X8')
#
# resp=inst.send_scpi_query(':SYST:ERR?')
#
# print(resp)

admin = TepAdmin() #required to control PXI module
sid = 8 #PXI slot WDS found
inst = admin.open_instrument(slot_id=sid)
resp = inst.send_scpi_query("*IDN?")
print('connected to: ' + resp) # Print *IDN
inst.send_scpi_cmd('*CLS; *RST')

#%%
sampleRateDAC = 8E9
inst.send_scpi_cmd(':INST:CHAN {0}'.format(1))
inst.send_scpi_cmd(':MODE DUC')
inst.send_scpi_cmd(':SOUR:INT X8')
inst.send_scpi_cmd(':FREQ:RAST {0}'.format(sampleRateDAC))
inst.send_scpi_cmd(':IQM ONE')


inst.send_scpi_cmd(':SOUR:NCO:CFR1 10E6')
inst.send_scpi_cmd(':SOUR:NCO:PHAS 0')

inst.send_scpi_cmd(':INST:CHAN {0}'.format(2))
inst.send_scpi_cmd(':MODE DUC')
inst.send_scpi_cmd(':SOUR:INT X8')
inst.send_scpi_cmd(':FREQ:RAST {0}'.format(sampleRateDAC))
inst.send_scpi_cmd(':IQM ONE')
inst.send_scpi_cmd(':SOUR:NCO:CFR1 10E6')


inst.send_scpi_cmd(':SOUR:INT X8')
inst.send_scpi_cmd(':FREQ:RAST {0}'.format(sampleRateDAC))


inst.send_scpi_query(':INT?')

resp=inst.send_scpi_query(':SYST:ERR?')

print(resp)