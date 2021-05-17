from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from inlet import *
from pylsl import StreamInlet, resolve_stream
import numpy as np

period = 10
data = [[0 for _ in range(4)] for _ in range(118)]
max_size = 128

fig, axs = plt.subplots(2, 2)

print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
print('Found an EEG stream!')

def animate(i):
	sample, _ = inlet.pull_sample()
	data.append(sample[:4])
	_data = np.transpose(np.array(data), (1,0))
	for i in range(4):
		axs[i//2, i%2].clear()
		axs[i//2, i%2].plot(range(len(_data[i])), _data[i])
		
	data.pop(0)
	# if len(data) >= max_size:
		# data = data[:max_size-period]
		

ani = FuncAnimation(plt.gcf(), animate, interval=200)
plt.show()