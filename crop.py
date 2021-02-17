import os.path as osp
import numpy as np
from PIL import Image
import os

def get_bbox(img):
    h, w = img.shape
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    r = rmax - rmin
    c = cmax - cmin
    x = np.round((rmax+rmin)/2)
    y = np.round((cmax+cmin)/2)
    x1 = x-256
    x2 = x+256
    y1 = y-256
    y2 = y+256
    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    return np.uint16(y1), np.uint16(x1), np.uint16(y2), np.uint16(x2)


def get_bbox_source(img):
    h, w = img.shape
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    r = rmax - rmin
    c = cmax - cmin
    x = np.round((rmax+rmin)/2)
    y = np.round((cmax+cmin)/2)
    x1 = x-256
    x2 = x+256
    y1 = y-256
    y2 = y+256
    print(x1, y1, x2, y2)
    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    return np.uint16(y1), np.uint16(x1), np.uint16(y2), np.uint16(x2)

if __name__ == '__main__':
    root = '/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/source'
    #list_path = '/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/target.txt'
    list_path = '/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/source.txt'
    new_root = '/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/source'
    img_ids = [i_id.strip() for i_id in open(list_path)]
    print(len(img_ids))
    '''
    image_path = '/mnt/ceph_fs/medai_data2/lolitazhang/my-master/eyes-master/path/to/RIM-ONE-r3/target/images'
    fileList = os.listdir(image_path)
    
    for name in fileList:
        if '_' in name:
            oldname = image_path + os.sep + name
            newname = image_path + os.sep + name.split('_')[0] + '.jpg'
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
    '''
    for name in img_ids:
        print(name)
        name_img = name.split('.')[0] + '.jpg'
        #name_img = name.split('.')[0] + '.png'
        name_label = name.split('.')[0] + '.bmp'
        print(name_img, name_label)
        image = Image.open(osp.join(root, "images/%s" % name_img)).convert('RGB')
        label = Image.open(osp.join(root, "labels/%s" % name_label))
       
        ind = {0: 1, 128: 1}
        s = np.asarray(label, np.float32)
        s_copy = np.zeros(s.shape, dtype=np.float32)
        for k, v in ind.items():
            s_copy[s == k] = v
        bbox = get_bbox_source(s_copy)
        image = image.crop(bbox)
        label = label.crop(bbox)
        save_image_path = osp.join(new_root, "disc_small/image")
        save_label_path = osp.join(new_root, "disc_small/mask")
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
        if not os.path.exists(save_label_path):
            os.makedirs(save_label_path)
        image.save(osp.join(save_image_path, name.split('.')[0] + '.png'))
        label.save(osp.join(save_label_path, name.split('.')[0] + '.png'))
