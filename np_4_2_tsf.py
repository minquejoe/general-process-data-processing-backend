import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import hpelm
from sklearn import metrics
from sklearn.svm import SVR


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=1, hiddens=6, layers=3):
        super(LSTM, self).__init__()
        # 不同的 batch 中 hidden_cell 不同
        self.batch_nums = 1
        self.layers = layers
        self.hiddens = hiddens

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hiddens,
            num_layers=layers,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=hiddens, out_features=1)

    def init_hidden(self):
        # 防止计算树图冲突，每次计算均重置
        # https://blog.csdn.net/SY_qqq/article/details/107384161
        # https://blog.csdn.net/qq_31375855/article/details/107568057
        # hx_0, cx_0
        self.hidden_cell = (
            torch.zeros(self.layers, self.batch_nums, self.hiddens),
            torch.zeros(self.layers, self.batch_nums, self.hiddens),
        )

    def forward(self, x):
        self.init_hidden()
        # x is input, size (batch_size, seq_len, input_size)
        x, self.hidden_cell = self.lstm(x, self.hidden_cell)
        # x is output, size (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]
        x = self.fc(x)
        x = x.view(1)
        return x


# 定义数据集合
class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.dataset = [x, y]

    def __len__(self):
        return np.shape(self.dataset)[-1]

    def __getitem__(self, i):
        inputs = self.dataset[0][i]
        outputs = self.dataset[1][i]

        return torch.tensor(inputs, dtype=torch.float32).view(-1, 1), torch.tensor(
            outputs, dtype=torch.float32
        )


class TSF_Data:
    """测试用数据集
    eg:
        x_pool, y_pool = TSF_Data().get_pool()
        train_loader, val_loader = TSF_Data().get_loader()
    """

    def __init__(self, data_len=1000, seq_len=50):
        x = np.linspace(0, 50, data_len)
        y = np.sin(x) + np.random.rand(data_len) / 2

        self.scaler = StandardScaler()
        y = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        seq_i = 0

        self.x_pool = []
        self.y_pool = []
        while seq_len + seq_i < len(x):
            self.x_pool.append(y[seq_i : seq_i + seq_len])
            self.y_pool.append(y[seq_i + seq_len])
            seq_i += 1

    def get_pool(self):
        return self.x_pool, self.y_pool

    def get_loader(self, train_ratio=0.8):
        self.dataset = Dataset(self.x_pool, self.y_pool)
        train_set, val_set = torch.utils.data.random_split(
            self.dataset,
            [
                int(len(self.dataset) * train_ratio),
                len(self.dataset) - int(len(self.dataset) * train_ratio),
            ],
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=1,
            #                                             collate_fn=collate_fn,
            shuffle=False,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=1,
            #                                             collate_fn=collate_fn,
            shuffle=False,
            drop_last=True,
        )
        return train_loader, val_loader


class TST_LSTM:
    """测试用LSTM网络
    注：测试模型路径 ./lstm_checkpoint.pt
    eg:
        pred, tgt = TST_LSTM().get_ans(val_loader)
    """

    def __init__(self, path="lstm_checkpoint.pt"):
        #  used the saved model parameters
        lstm = LSTM(input_size=1, hiddens=100, layers=4)
        lstm.load_state_dict(torch.load(path))
        self.lstm = lstm

    def get_ans(self, val_loader):
        self.lstm.eval()
        pred = []
        tgt = []
        for x, y in val_loader:
            pred.append(self.lstm(x).tolist())
            tgt.append(y.tolist())

        RMSE = metrics.mean_squared_error(pred, tgt) ** 0.5

        pred = np.array(pred).flatten().tolist()
        tgt = np.array(tgt).flatten().tolist()
        RMSE = RMSE.tolist()
        return pred, tgt, RMSE


class TSF_ELM:
    """测试用ELM
    eg:
        elm = TSF_ELM()
        pred, tgt, RMSE = elm.get_ans(x_pool, y_pool)
    """

    def __init__(self, L=30, seq_len=50):
        # Initialization
        self.L = L
        self.input_length = seq_len

    def get_ans(self, x, y, train_ratio=0.8):
        x = np.array(x)
        y = np.array(y)

        model = hpelm.ELM(self.input_length, 1)
        model.add_neurons(self.L, "sigm")

        # Train model
        model.train(x[: int(len(x) * train_ratio)], y[: int(len(y) * train_ratio)], "r")

        # Predict
        pred = model.predict(x[int(len(x) * train_ratio) :])

        # Calculate new RMSE
        RMSE = metrics.mean_squared_error(pred, y[int(len(y) * train_ratio) :]) ** 0.5
        
        pred = pred.flatten().tolist()
        tgt = y[int(len(y) * train_ratio) :].tolist()
        RMSE = RMSE.tolist()
        return [pred, tgt, RMSE]


class TSF_SVR:
    """测试用SVR
    eg:
        svr = TSF_SVR()
        pred, tgt, RMSE = svr.get_ans(x_pool, y_pool)
    """

    def __init__(self, gamma=0.001, C=10):
        self.gamma = gamma
        self.C = C

    def get_ans(self, x, y, train_ratio=0.8):
        x = np.array(x)
        y = np.array(y)

        svr = SVR(gamma=self.gamma, C=self.C)
        svr.fit(x[: int(len(x) * train_ratio)], y[: int(len(y) * train_ratio)])
        pred = svr.predict(x[int(len(x) * train_ratio) :])

        RMSE = metrics.mean_squared_error(pred, y[int(len(y) * train_ratio) :]) ** 0.5

        pred = pred.flatten().tolist()
        tgt = y[int(len(y) * train_ratio) :].tolist()
        RMSE = RMSE.tolist()
        return [pred, tgt, RMSE]
