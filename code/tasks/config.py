import lya
from lya import AttrDict
from pdb import set_trace as pause
import os.path as osp

__C = AttrDict()
cfg = __C

cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
cfg.DATA_DIR = osp.abspath(osp.join(cfg.ROOT_DIR, 'data'))


def load_config(cfg_file):
	update_config(lya.AttrDict.from_yaml(cfg_file))

def update_config(cfg_other):
    __C.update_dict(cfg_other)

