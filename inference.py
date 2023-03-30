import numpy as np
import tensorflow as tf
from utils.preset import preset
from utils import data_utils, train_utils, io_utils
from config import *
import tqdm
from losses import seg_loss
import cv2, glob, time, os

def main():
    model, _, _ = train_utils.get_model()
    start_time = time.time()
    frames = 0
    
    def inference_frame(frame):
        frame = tf.image.resize(frame, INPUT_SIZE)
        frame_gray = tf.image.rgb_to_grayscale(frame)[..., 0]
        inference = model(frame[None]/255.)
        inference = tf.cast(tf.argmax(inference[0], -1), tf.bool)
        output = tf.where(inference, tf.zeros_like(frame_gray), frame_gray)
        return output.numpy().astype(np.uint8)
    
    if INFERENCE_TYPE == 'image':
        image_files = glob.glob(IMAGE_INFERENCE_PATH + '*.png')
        output_dir = OUTPUT_DIR + INFERENCE_TYPE + '/' + IMAGE_INFERENCE_PATH.split('/')[-4]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_file in image_files:
            print(f'frames:{frames}/{len(image_files)}', end='\r', flush=True)
            image_name = '/' + image_file.split('\\')[-1]
            output_path = output_dir + image_name
            image = cv2.imread(image_file)[..., ::-1]
            inference = inference_frame(image)
            cv2.imwrite(output_path, inference)
            frames += 1
            
            
    elif INFERENCE_TYPE == 'video':
        cap = cv2.VideoCapture(VIDEO_INFERENCE_PATH)
        file_path = OUTPUT_DIR + INFERENCE_TYPE + '/' + VIDEO_INFERENCE_PATH.split('/')[-1].split('.')[0]
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(file_path + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, INPUT_SIZE[::-1])

        ret = True
        while(ret):
            print(f'frames:{frames}', end='\r', flush=True)
            ret, frame = cap.read()
            if ret:
                frame = frame[..., ::-1]
                inference = inference_frame(frame)
                inference = cv2.cvtColor(inference, cv2.COLOR_GRAY2BGR)
                writer.write(inference)
                frames += 1
  
        writer.release()
        cap.release()
        
    end_time = time.time()
    sec = end_time - start_time
    fps = frames / sec
    avg_sec = sec / frames
    
    print(f'total num of frames: {frames}')
    print(f'total inference time(s): {sec:.3f}')
    print(f'average inference time(s): {avg_sec:.3f}')
    print(f'average_fps(s): {fps:.3f}')
    
            

if __name__ == '__main__':
    preset()
    main()