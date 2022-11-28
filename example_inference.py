import torch
import numpy as np

from Model import iESPnet


FREQ_MASK_PARAM = 10
TIME_MASK_PARAN = 20
N_CLASSES = 1
learning_rate = 1e-3
batch_size = 128
epochs = 20
num_workers = 4

hparams = {
        "n_cnn_layers": 3,
    "n_rnn_layers": 3,
    "rnn_dim": [150, 100, 50],
    "n_class": N_CLASSES,
    "out_ch": [8,8,16],
    "dropout": 0.3,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_workers": num_workers,
    "epochs": epochs
}

model = torch.load(
    r"X:\Users\timon\Paper_iESPnet\Results\PIT-RNS9536\models\model_opt.pth",
    map_location=torch.device('cpu')
)

example_batch = r"X:\Users\timon\Paper_iESPnet\Data\PIT_RNS9536_20170927-1_E0.npy"
dat_np = np.load(example_batch, allow_pickle=True)
dat_np_arr = dat_np.item()["spectrogram"]

model.eval(dat_np_arr)

#### Application to .dat files:
b = open(example_batch, "rb").read()
ecog = np.frombuffer(b, dtype=np.int16)
ecog = ecog - 512
ecog = ecog.reshape([-1, 4])


dat = np.read_buffer(example_batch)
model.eval(dat)
