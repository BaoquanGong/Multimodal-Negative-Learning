#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from .bert import BertClf
from .bow import GloveBowClf
from .image import ImageClf
from .latefusion import MultimodalLateFusionClf
MODELS = {
    "bert": BertClf,
    "bow": GloveBowClf,
    "img": ImageClf,
    'latefusion':MultimodalLateFusionClf,
}

def get_model(args):
    return MODELS[args.model](args)
