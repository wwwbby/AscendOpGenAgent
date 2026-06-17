import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Extended ELU activation - unified float32 intermediate computation path
    for all dtypes to match Triton Ascend behavior.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, alpha=1.0, scale=1.0, input_scale=1.0):
        # Unified high-precision path: all dtypes use float32 intermediate
        x_f32 = x.float()
        alpha_f32 = torch.tensor(alpha, dtype=torch.float32, device=x.device)
        scale_f32 = torch.tensor(scale, dtype=torch.float32, device=x.device)
        input_scale_f32 = torch.tensor(input_scale, dtype=torch.float32, device=x.device)
        # Clamp exp argument to avoid overflow
        z = input_scale_f32 * x_f32
        z = torch.clamp(z, min=-11.0, max=11.0)
        result_f32 = torch.where(
            x_f32 > 0,
            scale_f32 * x_f32,
            alpha_f32 * scale_f32 * (torch.exp(z) - 1)
        )
        # Clamp to float16 max to avoid +Inf when converting back
        result_f32 = torch.clamp(result_f32, min=-65504.0, max=65504.0)
        return result_f32.to(x.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "13_Elu.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        alpha_info = inputs[1]
        scale_info = inputs[2]
        input_scale_info = inputs[3]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        dtype = dtype_map[x_info["dtype"]]
        x = torch.randn(x_info["shape"], dtype=dtype)
        alpha = float(alpha_info["value"])
        scale = float(scale_info["value"])
        input_scale = float(input_scale_info["value"])
        input_groups.append([x, alpha, scale, input_scale])
    return input_groups


def get_init_inputs():
    return []
