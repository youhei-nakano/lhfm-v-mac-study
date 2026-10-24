"""Own selected EMA + lean single example -> CPU prediction, no dataset download."""
import argparse
from pathlib import Path
import numpy as np
import torch
from architecture import Faithful
from stage1 import predict
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('example',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 saved=torch.load(a.checkpoint,weights_only=True,map_location='cpu');model=Faithful(mode=saved['mode']).eval();model.load_state_dict(saved['state_dict']);data=torch.from_numpy(np.load(a.example,allow_pickle=False)['frames'].copy());result=predict(model,data,'cpu');np.savez_compressed(a.output,prediction=result)
