import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dl.utils import show_image
show_image(dataset='fc_trainval')
show_image(dataset='tux')
show_image(dataset='action_trainval')