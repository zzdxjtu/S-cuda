import argparse
import os.path as osp
class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("--model", type=str, default='DeepLab',help="available options : DeepLab and VGG")
        parser.add_argument("--data-dir-target", type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/path/to/dataset/source', help="Path to the directory containing the source dataset.")
        parser.add_argument("--data-list-target", type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/refuge/level_0.5-0.7/select_0.1/jiao.txt', help="Path to the file listing the images in the source dataset.")
        parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.") 
        parser.add_argument("--num-classes", type=int, default=3, help="Number of classes for cityscapes.")
        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/path/to/snapshots/level_0.5-0.7/noise_labels_0.5/select_0.1/eyes_40', help="Where restore model parameters from.")
        parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")  
        parser.add_argument("--save", type=str, default='../correction/refuge/level_0.5-0.7/select_0.1/correction_label', help="Path to save result.")    
        parser.add_argument('--gt_dir', type=str, default = '/extracephonline/medai_data2/zhengdzhang/eyes/path/to/dataset/source/labels', help='directory which stores CityScapes val gt images')
        parser.add_argument('--devkit_dir', default='../correction/refuge/level_0.5-0.7/select_0.1', help='base directory of cityscapes')
        parser.add_argument('--p', type=list, default=[0.1, 0.1, 0.1], help='threshold for pseudo-label selection')
        parser.add_argument("--alpha", type=int, default=1, help="regularzation item.")         
        return parser.parse_args()
    
   
