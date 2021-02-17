import argparse
import os.path as osp
class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("--model", type=str, default='DeepLab',help="available options : DeepLab and VGG")
        parser.add_argument("--source", type=str, default='eyes',help="source dataset : gta5 or synthia")
        parser.add_argument("--target", type=str, default='eyes',help="target dataset : cityscapes")
        parser.add_argument("--batch-size", type=int, default=1, help="input batch size.")
        parser.add_argument("--num-workers", type=int, default=4, help="number of threads.")
        parser.add_argument("--data-dir", type=str, default='../path/to/dataset-new/source/eyesgan', help="Path to the directory containing the source dataset.")
        parser.add_argument("--data-list", type=str, default='../path/to/dataset-new/source.txt', help="Path to the file listing the images in the source dataset.")
        parser.add_argument("--data-dir-target", type=str, default='../path/to/dataset-new/target', help="Path to the directory containing the target dataset.")
        parser.add_argument("--data-list-target", type=str, default='../path/to/dataset-new/target.txt', help="Path to the file listing the images in the target dataset.")
        parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.")
        parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument("--learning-rate-D", type=float, default=1e-4, help="initial learning rate for discriminator.")
        parser.add_argument("--lambda-adv-target", type=float, default=0.001, help="lambda_adv for adversarial training.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--num-classes", type=int, default=3, help="Number of classes for cityscapes.")
        parser.add_argument("--num-steps", type=int, default=250000, help="Number of training steps.")
        parser.add_argument("--num-steps-stop", type=int, default=80000, help="Number of training steps for early stopping.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")
        parser.add_argument("--init-weights", type=str, default='../path/to/initial_weights/DeepLab_init.pth', help="initial model.")
        parser.add_argument("--restore-from", type=str, default='../path/to/snapshots/eyes_8/eyes_15000', help="Where restore model parameters from.")
        parser.add_argument("--save", type=str, default='../path/to/cityscapes/results_new', help="Path to save result.")
        parser.add_argument("--save-pred-every", type=int, default=1000, help="Save summaries and checkpoint every often.")
        parser.add_argument("--print-freq", type=int, default=10, help="print loss and time fequency.")
        parser.add_argument("--snapshot-dir", type=str, default='../path/to/snapshots/eyes_crop_8', help="Where to save snapshots of the model.")
        parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")
        parser.add_argument('--p', type=int, default=0.2, help='threshold for pseudo-label selection')
        parser.add_argument("--alpha", type=int, default=0.025, help="regularzation item.")
        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    
        # save to the disk
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')    
        
