from inlet import *
from model import *
import numpy as np

period = 10
data = []
max_size = 256
info = StreamInfo('Control', 'C', 3, 128//period, 'float32', 'ctrlid')
outlet = StreamOutlet(info)
for timestamp, sample in get_response():
		data.append(sample)
		if len(data) >= max_size:
			ctrl = predict(data[:max_size])
			data = data[:max_size-period]
			outlet.push_sample(ctrl)
