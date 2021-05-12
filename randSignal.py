import random
import time
import numpy as np 

from pylsl import StreamInfo, StreamOutlet


info = StreamInfo('BioSemi', 'EEG', 22, 128, 'float32', 'hello')

# next make an outlet
outlet = StreamOutlet(info)

print("now sending artificial data...")
time_sleep = 1/128.
while True:
		# make a new random 8-channel sample; this is converted into a
		# pylsl.vectorf (the data type that is expected by push_sample)
		mysample = [random.random() for _ in range(22)]
		# now send it and wait for a bit
		outlet.push_sample(mysample)
		time.sleep(time_sleep)