from config import *
from utils import io_utils
from models.deeplab import Model_ as Model

def step_lr_scheduler(epoch, warmup_iter):
    if epoch < 50:
        lr = LR
    elif epoch < 70:
        lr = LR*0.5
    elif epoch < 90:
        lr = LR*0.1
    elif epoch < 100:
        lr = LR*0.05
    else:
        lr = LR*0.01
    if warmup_iter < WARMUP:
        lr = lr / WARMUP * (warmup_iter+1)
    return lr

def poly_lr_scheduler(iter, max_iter, init_lr=LR, power=0.9):
    return init_lr*(1-(iter/(max_iter+1)))**power


def load_model(model):
    model.load_weights(CHECKPOINTS)
    saved_parameter = io_utils.read_model_info()
    return model, saved_parameter['epoch'], saved_parameter['max_loss']

def get_model():    
    if LOAD_CHECKPOINTS:
        return load_model(Model())
    return Model(), 0, 1e+50

def save_model(model, epoch, max_loss, valid=True):
    if valid:
        checkpoints = VALID_CHECKPOINTS_DIR + BACKBONE_TYPE
    else:
        checkpoints = TRAIN_CHECKPOINTS_DIR + BACKBONE_TYPE

    model.save_weights(checkpoints)
    io_utils.write_model_info(checkpoints, epoch, max_loss)