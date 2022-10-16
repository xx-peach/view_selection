from asyncore import write
import numpy as np
import os
import math

from config import config_parser
from utils.read_write_model import *

import warnings
warnings.filterwarnings("ignore")


param_type = {
    'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
    'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
    'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
    'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
    'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
    'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
    'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
    'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
    'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
    'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
    'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
}


def read_colmap_cameras(cameras, images):
    # intrinsic
    intrinsic = {}
    for camera_id, cam in cameras.items():
        params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
        if 'f' in param_type[cam.model]:
            params_dict['fx'] = params_dict['f']
            params_dict['fy'] = params_dict['f']
        i = np.array([
            [params_dict['fx'], 0, params_dict['cx']],
            [0, params_dict['fy'], params_dict['cy']],
            [0, 0, 1]
        ])
        intrinsic[camera_id] = i

    # extrinsic
    extrinsic = {}
    for image_id, image in images.items():
        e = np.zeros((4, 4))
        e[:3, :3] = qvec2rotmat(image.qvec)
        e[:3, 3] = image.tvec
        e[3, 3] = 1
        extrinsic[image_id] = e

    return intrinsic, extrinsic


def calc_score(inputs, extrinsic):
    """ modified version of mvsnet official implementation
        https://github.com/YoYo000/MVSNet/blob/master/mvsnet/colmap2mvsnet.py
    """
    i, j = inputs
    id_i = images[i+1].point3D_ids
    id_j = images[j+1].point3D_ids
    id_intersect = [it for it in id_i if it in id_j]
    cam_center_i = -np.matmul(extrinsic[i+1][:3, :3].transpose(), extrinsic[i+1][:3, 3:4])[:, 0]
    cam_center_j = -np.matmul(extrinsic[j+1][:3, :3].transpose(), extrinsic[j+1][:3, 3:4])[:, 0]
    score = 0
    for pid in id_intersect:
        if pid == -1:
            continue
        p = points3d[pid].xyz
        theta = (180 / np.pi) * np.arccos(np.dot(cam_center_i - p, cam_center_j - p) / np.linalg.norm(cam_center_i - p) / np.linalg.norm(cam_center_j - p))
        score += np.exp(-(theta - theta0) * (theta - theta0) / (2 * (sigma1 if theta <= theta0 else sigma2) ** 2))
    
    # set the nan to -1
    score = -1 if math.isnan(score) else score
    return score


def view_selection(ref_views, src_views, score):
    view_sel = []

    # generate the sorted pair one by one
    for i in range(len(ref_views)):
        sorted_score = np.argsort(score[i])[::-1][:-1]
        pair = [(src_views[k], score[ref_views[i], k]) for k in sorted_score]
        view_sel.append(pair)
    
    return view_sel


def write_pair(filename):
    with open(filename, 'w') as f:
        f.write('%d\n' % len(view_sel))
        for i, sorted_score in enumerate(view_sel):
            f.write('%d\n%d ' % (i, len(sorted_score)))
            for image_id, s in sorted_score:
                f.write('%d %f ' % (image_id, s))
            f.write('\n')



if __name__ == '__main__':
    args = config_parser()

    #TODO change the root_dir and colmap_bin to your own path
    source_dir = args.colmap_dir
    #TODO change the theta0, sigma1, sigma2 as your wish
    theta0, sigma1, sigma2 = args.theta0, args.sigma1, args.sigma2

    #TODO generate querying pair by your self
    # here I give an example of a set of pairs between every 49 view and every view in [0, 6, 12, 18, 24, 30, 36, 42, 48]
    ref_views = [i for i in range(49)]
    src_views = [i for i in range(0, 49, 6)]
    query_pairs = [(i, j) for i in ref_views for j in src_views]

    all_scores = np.zeros((len(ref_views), len(src_views)))

    for scene in args.select_scene_id:
        print(f">>> start to process scene {scene:03d}.")
        # refresh the target directory
        model_dir = os.path.join(source_dir, f'scan{scene}/sparse/0')
        # read the sparse model using colmap script
        cameras, images, points3d = read_model(model_dir, '.txt')
        # parse intrinsic and extrinsic from the sparse model
        intrinsic, extrinsic = read_colmap_cameras(cameras=cameras, images=images)

        # compute the similarity score for each query pair in current scene
        scores = [calc_score(pair, extrinsic=extrinsic) for pair in query_pairs]
        scores = np.array(scores).reshape(len(ref_views), len(src_views))
        all_scores += scores
        print(f">>> scene {scene:03d} done.\n")
    
    all_scores = all_scores / len(args.select_scene_id)

    # view selection
    view_sel = view_selection(ref_views=ref_views, src_views=src_views, score=all_scores)

    # write the pair file
    write_pair(filename=os.path.join(source_dir, 'pairs.txt'))

    os.system('rm -rf ./utils/__pycache__/')
    os.system('rm -rf ./.ipynb_checkpoints/')
    os.system('rm -rf ./__pycache__/')

