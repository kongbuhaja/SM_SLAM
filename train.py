import numpy as np
import tensorflow as tf
from utils.preset import preset
from utils import data_utils, train_utils, io_utils
from config import *
import tqdm
from losses import seg_loss

def main():   
    model, start_epoch, max_loss = train_utils.get_model()
    
    if OPTIMIZER == 'sgd':
        optimizer = tf.keras.optimizers.SGD()
    elif OPTIMIZER == 'adam':
        optimizer = tf.keras.optimizers.Adam()
    
    train_max_loss = valid_max_loss = max_loss
    
    train_writer = tf.summary.create_file_writer(LOGDIR)
    val_writer = tf.summary.create_file_writer(LOGDIR)
    
    dataset = data_utils.Dataset()
    train_dataset, train_length = dataset('train')
    valid_dataset, valid_length = dataset('val')
    
    train_dataset_length = train_length//BATCH_SIZE
    valid_dataset_length = valid_length//BATCH_SIZE
    
    global_iter = (start_epoch) * train_dataset_length
    max_iter = (EPOCHS) * train_dataset_length
    warmup_iter = 0
    
    for epoch in range(start_epoch, EPOCHS):
        train_iter, train_total_loss = 0, 0.
        train_tqdm = tqdm.tqdm(train_dataset, total=train_dataset_length, desc=f'train epoch {epoch}')
        for batch_data in train_tqdm:
            batch_images = batch_data[0]
            batch_labels = batch_data[1]           
            
            with tf.GradientTape() as train_tape:
                preds = model(batch_images, True)
                train_loss = model.loss(batch_labels, preds)
                gradients = train_tape.gradient(train_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                if LR_SCHEDULER == 'poly':
                    optimizer.lr.assign(train_utils.poly_lr_scheduler(global_iter, max_iter))
                if LR_SCHEDULER == 'step':
                    optimizer.lr.assign(train_utils.step_lr_scheduler(epoch, warmup_iter))
            
            global_iter += 1
            train_iter += 1
            warmup_iter += 1
            
            train_total_loss += train_loss
            _train_loss = train_total_loss / train_iter
            
            io_utils.write_summary(train_writer, global_iter, _train_loss, optimizer.lr.numpy())
            
            tqdm_text = f'lr={optimizer.lr.numpy():.7f}, average_loss={_train_loss:.7f}, iter_loss={train_loss:.7f}'
            train_tqdm.set_postfix_str(tqdm_text)
        if _train_loss < train_max_loss:
            train_max_loss = _train_loss
            train_utils.save_model(model, epoch, _train_loss, False)
            
        valid_iter, valid_total_loss = 0, 0.
        valid_tqdm = tqdm.tqdm(valid_dataset, total=valid_dataset_length, desc=f'valid epoch {epoch}')
        for batch_data in valid_tqdm:
            batch_images = batch_data[0]
            batch_labels = batch_data[1]
            
            preds = model(batch_images)
            valid_loss = model.loss(batch_labels, preds)
            
            valid_iter += 1
            
            valid_total_loss += valid_loss
            _valid_loss = valid_total_loss / valid_iter
            
            tqdm_text = f'#average_loss={_valid_loss:.5f}, #iter_loss={valid_loss:.5f}'
            valid_tqdm.set_postfix_str(tqdm_text)
            
        io_utils.write_summary(val_writer, epoch, _valid_loss)
        
        if _valid_loss < valid_max_loss:
            valid_max_loss = _valid_loss
            train_utils.save_model(model, epoch, _valid_loss)


if __name__ == '__main__':
    preset()
    main()