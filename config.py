import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    # general data path and colmap executable path
    parser.add_argument('--data_dir', type=str, default='../dtu', help='input dtu directory')
    parser.add_argument('--colmap_bin', type=str, default='/Applications/COLMAP.app/Contents/MacOS/colmap', help='colmap executable file path')
    # parameters for 'run_colmap.py'
    parser.add_argument('--height', type=int, default=1200, help='height of the image')
    parser.add_argument('--width', type=int, default=1600, help='width of the image')
    parser.add_argument('--colmap_scene_id', nargs='+', type=int, help='dtu scene names we want to reconstruct using colmap')
    # parameters for view selection
    parser.add_argument('--colmap_dir', type=str, default='./results', help='reconstructed colmap directory')
    parser.add_argument('--select_scene_id', nargs='+', type=int, help='dtu scene names we want to use for selection')
    parser.add_argument('--theta0', type=float, default=5, help='theta0')
    parser.add_argument('--sigma1', type=float, default=1, help='sigma1')
    parser.add_argument('--sigma2', type=float, default=10, help='sigma2')

    return parser.parse_args()
