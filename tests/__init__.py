import logging
import torch
logging.basicConfig(level=logging.DEBUG)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

torch.set_default_dtype(torch.float32)
