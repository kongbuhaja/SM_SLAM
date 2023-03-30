from config import *
import tensorflow as tf

def write_model_info(checkpoints, epoch, max_loss):
    with open(checkpoints + '.info', 'w') as f:
        text = f'epoch:{epoch}\n'
        text += f'max_loss:{max_loss}\n'
        f.write(text)

def read_model_info():
    saved_parameter = {}
    with open(CHECKPOINTS + '.info', 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, value = line[:-1].split(':')
            if key == 'epoch':
                saved_parameter[key] = int(value)
            else:
                saved_parameter[key] = float(value)
    return saved_parameter

def write_summary(writer, step, loss, lr=None):
    with writer.as_default():
        if lr != None:
            tf.summary.scalar('lr', lr, step=step)
            tf.summary.scalar('train_loss', loss, step=step)
            
        else:
            tf.summary.scalar("valid_loss", loss, step=step)
        
    writer.flush()