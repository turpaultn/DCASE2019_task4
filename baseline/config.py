import logging
import math
import sys


def create_logger(logger_name, log_file):
    '''
    Create a logger.
    The same logger object will be active all through out the python
    interpreter process.
    https://docs.python.org/2/howto/logging-cookbook.html
    Use   logger = logging.getLogger(logger_name) to obtain logging all
    through out
    '''
    logger = logging.getLogger(logger_name)
    # Remove the stdout handler
    logger_handlers = logger.handlers[:]
    for handler in logger_handlers:
        if handler.name == 'std_out':
            logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG)
    file_h = logging.FileHandler(log_file)
    file_h.setLevel(logging.DEBUG)
    file_h.set_name('file_handler')
    terminal_h = logging.StreamHandler(sys.stdout)
    terminal_h.setLevel(logging.INFO)
    terminal_h.set_name('stdout')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    tool_formatter = logging.Formatter(' %(levelname)s - %(message)s')
    file_h.setFormatter(formatter)
    terminal_h.setFormatter(tool_formatter)
    logger.addHandler(file_h)
    logger.addHandler(terminal_h)
    return logger


LOG = create_logger("baseline", "Baseline.log")

workspace = ""
# Dataset Paths
weak = 'dataset/metadata/train/weak.csv'
unlabel = 'dataset/metadata/train/unlabel_in_domain.csv'
synthetic = 'dataset/metadata/train/synthetic.csv'
validation = 'dataset/metadata/validation/validation.csv'
test2018 = 'dataset/metadata/validation/test_dcase2018.csv'
eval2018 = 'dataset/metadata/validation/eval_dcase2018.csv'

# config
# prepare_data
sample_rate = 44100
n_window = 2048
hop_length = 512
n_mels = 64
max_len_seconds = 10.
max_frames = math.ceil(max_len_seconds * sample_rate / hop_length)

f_min = 0.
f_max = 22050.

lr = 0.0001
initial_lr = 0.
beta1_before_rampdown = 0.9
beta1_after_rampdown = 0.5
beta2_during_rampdup = 0.99
beta2_after_rampup = 0.999


decay_step = 100*1000
decay_rate = 0.1
rampup_length = 10
rampdown_length = 1000


logit_distance_cost = 0.01

consistency_weak = 1.
consistency_strong = 1.
consistency_rampup = 10


median_window = 20

# Main
num_workers = 3
batch_size = 10
n_epoch = 3

checkpoint_epochs = 1

early_stopping = None
model_checkpoint = 1
save_best = True

conv_dropout = 0.3
dropout_non_recurrent = 0.3
activation = "relu"

best_threshold_weak = True
