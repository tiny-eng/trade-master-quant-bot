import numpy as np 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50,  output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, (h_n, _) = self.lstm3(out)    
        out = h_n[-1]  # last layer's hidden state at final timestep
        out = self.fc(out)
        return out

model_sp = LSTMModel()  
state= torch.load("model/SP_predict_best.pt" , weights_only= False )
model_sp.load_state_dict(state)

def predict(input , model):
    
    ##  input : List[List] size 17280 * 5 
    ##  output : Close Price

    scaler = MinMaxScaler(feature_range=(0,1))

    input = scaler.fit_transform(np.array(input))
    input = torch.tensor(input , dtype=torch.float32)

    test_predict = model(input)
    test_predict = test_predict.detach().numpy()

    new_array = np.zeros((17280 , 5))
    new_array[0 , 3] = test_predict[0]
    test_predict = scaler.inverse_transform(new_array)
    return test_predict[0 , 3]


# ["Open" ,"High" ,"Low" ,"Close" ,"Volume"]  5 min period of 60 days data
#  60 * 24 * 60 / 5 = 17280s
input = [[1,2,3,4 + j * 0.001,5] for j in range(17280)] 
print(predict(input, model_sp))