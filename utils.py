import time
import torch.nn.functional as F

from datetime import datetime

def prepad_time(t):
    if len(str(t)) == 1:
        return "0" + str(t)
    else:
        return str(t)

def get_file_time(t0, args):
    dt0 = datetime.fromtimestamp(t0)
    file_name = "{}_{}_{}_{}_{}{}{}_{}{}{}".format(
        args.dataset,
        args.noise_type,
        args.sample_split,
        args.ssl,
        dt0.year,
        prepad_time(dt0.month),
        prepad_time(dt0.day),
        prepad_time(dt0.hour),
        prepad_time(dt0.minute),
        prepad_time(dt0.second),
    )
    return file_name
