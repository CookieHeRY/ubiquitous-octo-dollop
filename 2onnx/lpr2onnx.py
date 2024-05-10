import torch
from model.LPRNet import build_lprnet

lprnet = build_lprnet(lpr_max_len=8, phase=64, class_num=68,
                      dropout_rate=0.5)
device = torch.device("cuda:0" if torch.cuda else "cpu")
lprnet.to(device)
print("Successful to build net work!")
input = (1, 3, 24, 94)
input_data = torch.randn(input).to(device)
torch.onnx.export(lprnet, input_data, "lpr.onnx")