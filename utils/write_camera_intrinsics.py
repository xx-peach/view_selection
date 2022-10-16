import argparse
import numpy as np
import sys
sys.path.append('.')
import database

def main():
    parser = argparse.ArgumentParser(description='modify colmap database')
    parser.add_argument('--db_path', type=str, required=True)
    parser.add_argument('--camera_path', type=str, required=True)
    args = parser.parse_args()

    # open the database that we are going to modify
    db = database.COLMAPDatabase.connect(args.db_path)

    # define the correspondance from model type -> int
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

    # modify the database
    ids, models, widths, heights, params = [], [], [], [], []
    with open(args.camera_path, 'r') as file:
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

if __name__ == '__main__':

    main()
