import os
import argparse


LEAVE_PRETRAIN_FILE_CONDITIONS = [
    '_60ep', '_300ep'
]

LEAVE_FINETUNE_FILE_CONDITIONS = [
    '_1000ep'
]

def is_delete_file(filename, is_pretrain=True):
    if is_pretrain:
        return not any(condition in filename for condition in LEAVE_PRETRAIN_FILE_CONDITIONS)
    else:
        return not any(condition in filename for condition in LEAVE_FINETUNE_FILE_CONDITIONS)


def is_pth_file(filename):
    return filename.endswith('.pth')


def is_pretrain_dir(root):
    if 'pretrain' in root:
        return True
    elif 'finetune' in root:
        return False
    

def main(args):
    for dataroot in args.dataroots:
        for root, dnames, fnames in os.walk(dataroot, followlinks=True):
            for fname in fnames:
                if is_pth_file(fname):
                    if is_delete_file(fname, is_pretrain_dir(root)):
                        os.remove(os.path.join(root, fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroots', type=str, nargs='+', required=True)
    args = parser.parse_args()

    main(args)
