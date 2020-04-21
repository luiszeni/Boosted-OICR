# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Utilities for logging."""
from collections import deque
from email.mime.text import MIMEText
import json
import logging
import numpy as np
import smtplib
import sys

from tasks.config import cfg
from utils.smoothed_value import SmoothedValue

from pdb import set_trace as pause
# Print lower precision floating point values than default FLOAT_REPR
# Note! Has no use for json encode with C speedups
json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')


def log_json_stats(stats, sort_keys=True):
    print('json_stats: {:s}'.format(json.dumps(stats, sort_keys=sort_keys)))


def log_stats(stats, misc_args):
    """Log training statistics to terminal"""
    if hasattr(misc_args, 'epoch'):
        lines = "[%s][%s][Epoch %d][Iter %d / %d]\n" % (
            misc_args.run_name, misc_args.cfg_filename,
            misc_args.epoch, misc_args.step, misc_args.iters_per_epoch)
    else:
        lines = "[%s][%s][Step %d / %d]\n" % (
            misc_args.run_name, misc_args.cfg_filename, stats['iter'], cfg.SOLVER.MAX_ITER)

    lines += "\t\tloss: %.6f, lr: %.6f time: %.6f, eta: %s\n" % (
        stats['loss'], stats['lr'], stats['time'], stats['eta']
    )

    if stats['extras'] is not None:
        extra_line = '\t\t'
        extras = stats['extras']
        for k, v in extras.items():
            extra_line += "%s: %.6f," % (k,v)
        lines +=  extra_line + '\n'

    if stats['head_losses']:
        lines += "\t\t" + ", ".join("%s: %.6f" % (k, v) for k, v in stats['head_losses'].items()) + "\n"
    print(lines[:-1])  # remove last new line




def send_email(subject, body, to):
    s = smtplib.SMTP('localhost')
    mime = MIMEText(body)
    mime['Subject'] = subject
    mime['To'] = to
    s.sendmail('detectron', to, mime.as_string())


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger