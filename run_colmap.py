import os
import numpy as np

from config import config_parser

# from https://github.com/colmap/colmap/tree/dev/scripts/python
from utils.read_write_model import rotmat2qvec
from utils.database import COLMAPDatabase
from utils import database
from utils.read_write_model import read_model, write_model


camModelDict = {
    'SIMPLE_PINHOLE': 0,
    'PINHOLE': 1,
    'SIMPLE_RADIAL': 2,
    'RADIAL': 3,
    'OPENCV': 4,
    'FULL_OPENCV': 5,
    'SIMPLE_RADIAL_FISHEYE': 6,
    'RADIAL_FISHEYE': 7,
    'OPENCV_FISHEYE': 8,
    'FOV': 9,
    'THIN_PRISM_FISHEYE': 10
}


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))             # (4, 4)
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
    intrinsics_[:3, :3] = intrinsics                    # (4, 4)
    return intrinsics_, extrinsics


def load_cam_info(root_dir, scene_name, id_list):
    all_intrinsics, all_extrinsics = [], []
    # read the intrinsic and extrinsic matrix for each view in the `id_list`
    for vid in id_list:
        proj_mat_filename = os.path.join(root_dir, f'scan{scene_name}/cams/{vid:08d}_cam.txt')
        intrinsic, extrinsic = read_cam_file(proj_mat_filename)
        all_intrinsics.append(intrinsic)
        all_extrinsics.append(extrinsic)
    
    return all_intrinsics, all_extrinsics


def create_image_dir(target, source, scene_id, id_list, all_extrinsics):
    # create the image directory
    os.makedirs(f'{target}/images', exist_ok=True)
    
    data_list = []
    for i in id_list:
        # copy the image from original dataset to the colmap project directory
        os.system(f'cp {source}/scan{scene_id}/images/{i:08d}.jpg {target}/images')
        # transform the extrinsic matrix to qvec form
        rt = all_extrinsics[i]
        rt = np.linalg.inv(rt)
        r = rt[:3, :3]
        t = rt[:3,  3]
        # generate the data form for the `images.txt`
        q = rotmat2qvec(r)
        data = [i+1, *q, *t, 1, f'{i:08d}.jpg']
        data = [str(_) for _ in data]
        data = ' '.join(data)
        data_list.append(data)

    return data_list


def create_model_dir(target, width, height, all_intrinsics, data_list):
    # create the model directory
    os.makedirs(f'{target}/model/', exist_ok=True)
    # create the empty `point3D.txt` file
    os.system(f'touch {target}/model/points3D.txt')
    # create the `cameras.txt` and write into it
    with open(f'{target}/model/cameras.txt', 'w') as f:
        for i, intrinsic in enumerate(all_intrinsics):
            f.write(f'{i+1} PINHOLE {width} {height} {intrinsic[0][0]} {intrinsic[1][1]} {intrinsic[0][2]} {intrinsic[1][2]}')
            f.write('\n')
    # create the `images.txt` and write into it
    with open(f'{target}/model/images.txt', 'w') as f:
        for data in data_list:
            f.write(data)
            f.write('\n\n')


def update_database(target, camModelDict):
    # open the database
    db = COLMAPDatabase.connect(f'{target}/database.db')
    # modify the database
    ids, models, widths, heights, params = [], [], [], [], []
    with open(f'{target}/model/cameras.txt', 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 1):
            l = lines[i].split(' ')
            id = int(l[0])                  # camera id
            model = camModelDict[l[1]]      # camera model type
            width, height = int(l[2]), int(l[3])
            param = np.array([np.float64(x) for x in l[-4:]]).astype(np.float64)
            ids.append(id), models.append(model), widths.append(width), heights.append(height), params.append(param)
            db.update_camera(model, width, height, param, id)
    # commit the changes
    db.commit()
    # check whether the modification is successful or not
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0, len(ids), 1):
        id, model, width, height, param, prior = next(rows)
        param = database.blob_to_array(param, np.float64)
        assert id == ids[i]
        assert model == models[i] and width == widths[i] and height == heights[i]
        assert np.allclose(param, params[i])
    # close the database
    db.close()


def points3d_to_ply(in_points3d, out_ply_file='makeply.ply'):

    xyzs, rgbs = [], []
    with open(in_points3d, 'r') as file:
        lines = file.readlines()
        for line in lines[3:]:
            line = line.rstrip().split(' ')
            xyzs.append(np.array([float(line[1]), float(line[2]), float(line[3])]))
            rgbs.append(np.array([  int(line[4]),   int(line[5]),   int(line[6])]))

    with open(out_ply_file, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(len(xyzs)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyzs)):
            r, g, b = rgbs[i]
            x, y, z = xyzs[i]
            f.write('{} {} {} {} {} {}\n'.format(x, y, z, r, g, b))



if __name__ == '__main__':
    args = config_parser()

    #TODO change the root_dir and colmap_bin to your own path
    source_dir = args.data_dir
    colmap_bin = args.colmap_bin
    #TODO change the image width and height according to the specific dtu version your are using
    width, height = args.width, args.height

    # load camera intrinsic and extrinsic for all the views
    id_list = [i for i in range(49)]
    all_intrinsics, all_extrinsics = load_cam_info(root_dir=source_dir, scene_name=1, id_list=id_list)

    for scene in args.colmap_scene_id:
        # refresh the target directory
        target_dir = os.path.join('./results', f'scan{scene}')
        os.system(f'rm -rf {target_dir}')
        print(f'>>> start to process scene {scene:03d}.')

        # create the image directory
        image_info = create_image_dir(target=target_dir, source=source_dir, scene_id=scene, id_list=id_list, all_extrinsics=all_extrinsics)
        # create the reference model directory
        create_model_dir(target=target_dir, width=width, height=height, all_intrinsics=all_intrinsics, data_list=image_info)
        # create the logging directory
        os.makedirs(f'{target_dir}/logs/', exist_ok=True)

        # colmap feature extracting
        os.system(f'{colmap_bin} feature_extractor --database_path {target_dir}/database.db --image_path {target_dir}/images > {target_dir}/logs/feature_extractor_output.txt')
        print(f'>>> colmap feature_extractor for scene {scene:03d} done.')
        
        # colmap exhaustive matcher
        update_database(target=target_dir, camModelDict=camModelDict)
        os.system(f'{colmap_bin} exhaustive_matcher --database_path {target_dir}/database.db > {target_dir}/logs/exhaustive_matcher_output.txt')
        print(f'>>> colmap exhaustive_matcher for scene {scene:03d} done.')
        
        # colmap mapper
        os.makedirs(f'{target_dir}/sparse/', exist_ok=True)
        # os.system(f'{colmap_bin} point_triangulator --database_path {target_dir}/database.db --image_path {target_dir}/images --input_path {target_dir}/model --output_path {target_dir}/sparse > {target_dir}/logs/point_triangulator_output.txt')
        os.system(f'{colmap_bin} mapper --database_path {target_dir}/database.db --image_path {target_dir}/images --output_path {target_dir}/sparse > {target_dir}/logs/mapper_output.txt')
        print(f'>>> colmap mapper for scene {scene:03d} done.')

        # transform the results from '.bin' format to '.txt' format
        cameras, images, points3D = read_model(path=f'{target_dir}/sparse/0/', ext='.bin')
        cameras, images, points3D = write_model(cameras, images, points3D, path=f'{target_dir}/sparse/0', ext='.txt')

        # transform the reconstructed 3D points to ply file
        points3d_to_ply(in_points3d=f'{target_dir}/sparse/0/points3D.txt', out_ply_file=f'{target_dir}/makeply.ply')
        print(f'>>> ply file generated for scene {scene:03d}\n')

    os.system('rm -rf ./utils/__pycache__/')
    os.system('rm -rf ./.ipynb_checkpoints/')
    os.system('rm -rf ./__pycache__/')

