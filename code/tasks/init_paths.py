"""Add {PROJECT_ROOT}/code. to PYTHONPATH
""" 

import os.path as osp
import sys
from pdb import set_trace as pause

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
add_path(this_dir)
