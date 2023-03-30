EPOCHS = 30
BATCH_SIZE = 2
LR = 0.007
DTYPE = 'carla'
EPS = 1e-7
INPUT_SIZE = [512, 1024]

# sgd, adam
OPTIMIZER = 'adam'

# poly, step
LR_SCHEDULER = 'poly'

if OPTIMIZER == 'adam':
    WARMUP = 300
else:
    WARMUP = 0
    
# sigmoid, softmax
NORM = 'sigmoid'

# ce, bce, mse
LOSS = 'bce'

# xception, resnet
BACKBONE_TYPE = 'xception'
CHECKPOINTS_DIR = 'checkpoints/' + DTYPE + '/'
TRAIN_CHECKPOINTS_DIR = CHECKPOINTS_DIR + BACKBONE_TYPE + '/train/'
VALID_CHECKPOINTS_DIR = CHECKPOINTS_DIR + BACKBONE_TYPE + '/valid/'
CHECKPOINTS = VALID_CHECKPOINTS_DIR + BACKBONE_TYPE
LOAD_CHECKPOINTS = False


# log config
LOGDIR = 'logs/' + DTYPE + '/' + BACKBONE_TYPE + '/'

# inference config
OUTPUT_DIR = 'outputs/'
# photo, video
INFERENCE_TYPE = 'video'
IMAGE_INFERENCE_PATH = 'data/carla/val/front/'
VIDEO_INFERENCE_PATH = 'data/carla/val/video.avi'
# data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/

if DTYPE == 'cityscapes':
    NUM_CLASSES = 34
elif 'carla' in DTYPE:
    NUM_CLASSES = 9