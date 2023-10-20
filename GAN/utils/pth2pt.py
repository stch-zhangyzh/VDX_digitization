import torch
import torchvision
from custom_classical_models import *

pth_path = "C:\\work_local\\TBGan\\runs\\CNN_sec_OptunaResult_cuda\\s0\\checkpoints\\epoch_199.pth"

model = Optuna_Gen(input_size=50, hidden_layers_size=[106,78,14], batch_norm=1)

model.load_state_dict(torch.load(pth_path))
model.eval()
example = torch.rand(1, 3, 320, 480)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("predictor.pt")