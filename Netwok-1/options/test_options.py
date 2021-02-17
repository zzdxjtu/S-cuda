import argparse
import os.path as osp
class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("--model", type=str, default='DeepLab',help="available options : DeepLab and VGG")
        #parser.add_argument("--data-dir-target", type=str, default='/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/source')
        parser.add_argument("--data-dir-target", type=str, default='/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/test', help="Path to the directory containing the source dataset.")
        #parser.add_argument("--data-list-target", type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/disc_small/crop.txt')
        parser.add_argument("--data-list-target", type=str, default='/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/test.txt', help="Path to the file listing the images in the source dataset.")
        parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.") 
        parser.add_argument("--num-classes", type=int, default=3, help="Number of classes for cityscapes.")
        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/n1/snapshots/level_0.5-0.7/noise_labels_0.1/select_0.9/eyes_360', help="Where restore model parameters from.")
        #parser.add_argument("--restore-from", type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/path/to/snapshots/level_0.5-0.7/noise_labels_0.5/update_select_0.1/eyes_40', help="Where restore model parameters from.")
        parser.add_argument("--set", type=str, default='val', help="choose adaptation set.")  
        #parser.add_argument("--save", type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/path/to/result/level_0.2-0.3/noise_labels_0.9/select_0.1_scratch', help="Path to save result.")
        parser.add_argument("--save", type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/result/level_0.5-0.7/noise_labels_0.1/select_0.9')    
        #parser.add_argument('--gt_dir', type=str, default = '/extracephonline/medai_data2/zhengdzhang/eyes/qikan/path/to/test/labels', help='directory which stores CityScapes val gt images')
        parser.add_argument('--gt_dir', type=str, default= '/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/test/labels')
        parser.add_argument('--devkit_dir', default='/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new', help='base directory of cityscapes')
        #parser.add_argument('--devkit_dir', default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/disc_small')         
        return parser.parse_args()
    
   
