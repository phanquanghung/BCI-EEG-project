import torch
import torch.nn as nn

thres = .5  

class NNet(nn.Module):
    def __init__(self, in_channels=22, out_channels=3):
        super(NNet, self).__init__()
        self.hidden = 16
        self.net1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, in_channels, 5, padding=2),  # 128
            nn.Conv1d(in_channels, self.hidden, 4, stride=4),  # 32
            nn.LeakyReLU(0.01),
            nn.Conv1d(self.hidden, self.hidden, 7, padding=3), # 32
        )
        self.net2 = nn.Sequential(
            self.__block(self.hidden, self.hidden),
            self.__block(self.hidden, self.hidden)  # 8
        )
        
        self.mid = nn.Sequential(
            self.__block(self.hidden, self.hidden)
        ) # 4
        self.final = nn.Sequential(
            nn.Linear(192, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, out_channels),
            nn.Sigmoid()
        )
        
    def __block(self, inchannels, outchannels):
        return nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Conv1d(inchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(outchannels),
            nn.Conv1d(outchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.01)
        )
    
    def forward(self, x):
        x = self.net1(x)
        y1 = self.net2(x)
        y = self.mid(y1)
        y = torch.cat((x[..., -4:], y, y1[..., -4:]), dim=-1).view(x.shape[0], -1)
        return self.final(y)

PATH = 'model.pt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Model running on {}'.format(device))				
nnet = NNet()
nnet.load_state_dict(torch.load(PATH),map_location=device)
nnet.to(device)
nnet.eval()

def predict(input):
	with torch.no_grad():
		input = np.array(input, dtype=np.float32).reshape(1, 256, 22)
		input = torch.from_numpy(input).to(device)
		output = nnet(input).squeeze().detach().cpu().numpy()
	output = output > thres
	return output