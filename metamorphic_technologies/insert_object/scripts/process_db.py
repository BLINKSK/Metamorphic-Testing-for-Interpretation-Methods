# this is the whole workflow
# three modules

import os
from os import walk
import os.path
import sys

sys.path.append('../')
import object_refinement.refinement_process as RP


def init_object_refinement():
    # refinement; clustering and so on
    # basically even if we skip this one, we still can
    # proceed the following steps, right?
    target_dir = "../object_pool/"
    image_dir = "../image_pool/"
    RP.init(target_dir, image_dir)


init_object_refinement()