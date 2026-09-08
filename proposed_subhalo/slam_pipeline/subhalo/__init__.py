# Library modules only. NOTE: database.py is a standalone example SCRIPT (it
# runs at import time — builds a .sqlite, queries the aggregator, prints), so it
# is deliberately NOT imported here. Run it directly if you want it:
#   python slam_pipeline/subhalo/database.py
from . import detect
from . import tiling
from . import loaders
