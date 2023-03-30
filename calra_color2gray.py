import os, cv2
import numpy as np

# color_map = dict()
# color_map['unlabeld'] = [152, 251, 152] #0 10025880
# color_map['road'] = [128, 64, 128] #1 8405120
# color_map['sidewalk'] = [244, 35, 232] #2 15999976
# color_map['building'] = [70, 70, 70] #3 4605510
# color_map['wall'] = [102, 102, 156] #4
# color_map['fence'] = [190, 153, 153] #5
# color_map['pole'] = [153, 153, 153] #6
# color_map['traffic light'] = [250, 170, 30] #7
# color_map['traffic sign'] = [220, 220, 0] #8
# color_map['vegetation'] = [107, 142, 35] #9
# color_map['terrain'] = [152, 151, 152] #10
# color_map['sky'] = [70, 130, 180] #11
# color_map['pedestrian'] = [220, 20, 60] #12 ㅁ
# color_map['rider'] = [220, 0, 0] #13 ㅁ
# color_map['Car'] = [0, 0, 142] #14 ㅁ
# color_map['truck'] = [0, 0, 70] #15 ㅁ
# color_map['bus'] = [0, 60, 100] #16 ㅁ
# color_map['train'] = [0, 80, 100] #17 ㅁ
# color_map['motorcycle'] = [0, 0, 230] #18 ㅁ
# color_map['bicycle'] = [119, 11, 32] #19 ㅁ
# color_map['static'] = [110, 190, 160] #20
# color_map['dynamic'] = [170, 120, 50] #21
# color_map['other'] = [55,  90, 80] #22
# color_map['water'] = [45,  60, 150] #23
# color_map['road line'] = [157, 234, 50] #24
# color_map['ground'] = [81,   0, 81] #25
# color_map['bridge'] = [150, 100, 100] #26
# color_map['rail track'] = [230, 150, 140] #27
# color_map['guard rail'] = [180, 165, 180] #28

# color_map_ = dict()
# color_map_[14423100] = 1 # = [220, 20, 60] pedestrian
# color_map_[14417920] = 2 # = [220, 0, 0] rider
# color_map_[142] = 3 # = [0, 0, 142] #14 Car
# color_map_[70] = 4 # = [0, 0, 70] #15 truck
# color_map_[15460] = 5 # = [0, 60, 100] bus
# color_map_[20580] = 6 # = [0, 80, 100] train
# color_map_[230] = 7 # = [0, 0, 230] motorcycle
# color_map_[7801632] = 8 # = [119, 11, 32] bicycle

color_map_=dict()
color_map_[0]=0
color_map_[220020060]=1 #pedestrian
color_map_[220000000]=2 #rider
color_map_[142]=3 #Car
color_map_[70]=4 #truck
color_map_[60100]=5 #bus
color_map_[80100]=6 #train
color_map_[230]=7 #motorcycle
color_map_[119011032]=8 #bicycle
# color_map_[107142035]=9 #vegetation

def new_concat(image):
    return (image[..., 0]*1000 + image[..., 1]) *1000 + image[..., 2]

def cvt_images(dir_path):
    save_path = dir_path[:-1] + '_gray/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    image_files = os.listdir(dir_path)
    images = []
    for i, image_file in enumerate(image_files):
        print(f'{i}/{len(image_files)}', end='r', flush=True)
        images += [cv2.imread(dir_path + image_file)[..., ::-1]]
    
    images = new_concat(np.array(images, np.int32))
    masks = np.zeros_like(images, np.int32)
    for key, value in color_map_.items():
        masks += np.where(images==key, value, 0)        

    for mask, image_file in zip(masks.astype(np.uint8), image_files):
        cv2.imwrite(save_path + image_file, mask)
        
cvt_images('data/carla/val/semantic/')