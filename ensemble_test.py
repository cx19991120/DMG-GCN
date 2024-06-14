import argparse
import pickle
import os
import math
import numpy as np
from tqdm import tqdm

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)
    parser.add_argument('--main-dir',
                        help='')
    parser.add_argument('--left',
                        type=str2bool,
                        default=True)
    parser.add_argument('--middle',
                        type=str2bool,
                        default=True)
    parser.add_argument('--right',
                        type=str2bool,
                        default=True)



    arg = parser.parse_args()

    dataset = arg.dataset

    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError


    dir_cnt = 0
    if arg.left:
        with open(os.path.join(arg.main_dir, 'ctrgcn_joint/', 'epoch1_test_score.pkl'), 'rb') as r1:
            r1 = list(pickle.load(r1).items())
        with open(os.path.join(arg.main_dir, 'ctrgcn_bone/', 'epoch1_test_score.pkl'), 'rb') as r2:
            r2 = list(pickle.load(r2).items())
        dir_cnt += 2

    if arg.middle:
        with open(os.path.join(arg.main_dir, 'ctrgcn_joint_body_part_middle/', 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
        with open(os.path.join(arg.main_dir, 'ctrgcn_bone_body_part_middle/', 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())
        dir_cnt += 2

    if arg.right:
        if 'ntu' in arg.dataset:
            with open(os.path.join(arg.main_dir, 'ctrgcn_joint_body_part_right/' 'epoch1_test_score.pkl'), 'rb') as r5:
                r5 = list(pickle.load(r5).items())
            with open(os.path.join(arg.main_dir, 'ctrgcn_bone_body_part_right/', 'epoch1_test_score.pkl'), 'rb') as r6:
                r6 = list(pickle.load(r6).items())
            dir_cnt += 2
        elif 'UCLA' in arg.dataset:
            with open(os.path.join(arg.main_dir, 'ctrgcn_joint/' 'epoch1_test_score.pkl'), 'rb') as r5:
                r5 = list(pickle.load(r5).items())
            with open(os.path.join(arg.main_dir, 'ctrgcn_bone/', 'epoch1_test_score.pkl'), 'rb') as r6:
                r6 = list(pickle.load(r6).items())
            dir_cnt += 2

    right_num = total_num = right_num_5 = 0
    norm = lambda x: x / np.linalg.norm(x)

    if dir_cnt == 6:
        arg.alpha = [0.65, 0.64, 0.4, 0.4, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            _, r55 = r5[i]
            _, r66 = r6[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3] + r55 * arg.alpha[4]+ r66 * arg.alpha[5]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    elif dir_cnt == 4:
        r = None
        arg.alpha = [0.62, 0.6, 0.4, 0.4, 0.6, 0.6]
        for i in tqdm(range(len(label))):
            l = label[i]
            if arg.left:
                _,r11 = r1[i]
                _,r22 = r2[i]
                r = r11 * arg.alpha[0] + r22 * arg.alpha[1]
            if arg.middle:
                _,r33 = r3[i]
                _,r44 = r4[i]
                r = r + r33 * arg.alpha[2] + r44 * arg.alpha[3] if r is not None else r33 * arg.alpha[2] + r44 * arg.alpha[3]
            if arg.right:
                _,r55 = r5[i]
                _,r66 = r6[i]
                r = r + r55 * arg.alpha[5] + r66 * arg.alpha[6] if r is not None else r55 * arg.alpha[5] + r66 * arg.alpha[6]

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    elif dir_cnt == 2:
        r = None
        arg.alpha = [0.62, 0.6, 0.4, 0.4, 0.6, 0.6]
        for i in tqdm(range(len(label))):
            l = label[i]
            if arg.left:
                _,r11 = r1[i]
                _,r22 = r2[i]
                r = r11 * arg.alpha[0] + r22 * arg.alpha[1]
            if arg.middle:
                _,r33 = r3[i]
                _,r44 = r4[i]
                r = r + r33 * arg.alpha[2] + r44 * arg.alpha[3] if r is not None else r33 * arg.alpha[2] + r44 * arg.alpha[3]
            if arg.right:
                _,r55 = r5[i]
                _,r66 = r6[i]
                r = r + r55 * arg.alpha[5] + r66 * arg.alpha[6] if r is not None else r55 * arg.alpha[5] + r66 * arg.alpha[6]

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    print(dir_cnt)
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))