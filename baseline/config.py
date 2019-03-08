import math

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
weight_decay_during_rampup = 0.99
weight_decay_after_rampup = 0.999

decay_step = 100*1000
decay_rate = 0.1

logit_distance_cost = 0.01

train_iter_count = 20000

rampup_length = 0 # int(0.5 * train_iter_count)
rampdown_length = 0 # int(0.1 * train_iter_count)

max_consistency_cost = 100.
max_learning_rate = 0.001
consistency_weak = 10
consistency_strong = 10
consistency_rampup = 10






median_window = 20

# Main
num_workers = 12
batch_size = 32
n_epoch = 100

checkpoint_epochs = 1

early_stopping = None
model_checkpoint = 1
save_best = True

dropout = 0.5
activation = "glu"

best_threshold_weak = True
