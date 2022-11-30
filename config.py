
EXPERIMENT_NAME = 'preload_params'
KERNEL_CHOICE = 'Matern32'

LOAD_PARAMS = True
SAVE_PARAMS = False

assert not (LOAD_PARAMS and SAVE_PARAMS), "Cannot save and load at the same time"

DEBUG = False
VERBOSE = False or DEBUG

