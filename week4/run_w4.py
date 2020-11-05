import os
import sys
from glob import glob
import argparse
import pickle

# import week4.retrieval as retrieval
# import week4.evaluation as evaluation

def parse_args(args=sys.argv[2:]):
    parser = argparse.ArgumentParser(description='CBIR: Content Based Image Retrieval. MCV-M1-Project, Team 3')

    parser.add_argument('bbdd_path', type=str,
                        help='absolute/relative path to the bbdd dataset')

    parser.add_argument('query_path', type=str,
                        help='absolute/relative path to the query dataset')

    parser.add_argument('--test', action='store_true',
                        help='using a test dataset, so no groundtruth is provided')

    parser.add_argument('--map_k', type=lambda s: [int(item) for item in s.split(',')], default=[5],
                        help='retrieve K number/s of images')

    parser.add_argument('--remove_bg', action='store_true',
                        help='remove background from images in order to extract paintings')

    parser.add_argument('--max_paintings', type=int, default=1,
                        help='maximum number of paintings to extract from an image after removing the background')

    parser.add_argument('--remove_text', action='store_true',
                        help='remove text bounding boxes from images')

    parser.add_argument('--use_color', action='store_true',
                        help='use color descriptor')

    parser.add_argument('--use_texture', action='store_true',
                        help='use texture descriptor/s')

    parser.add_argument('--use_text', action='store_true',
                        help='use text descriptor')

    parser.add_argument('--color_descriptor', type=str, default='3d_rgb_blocks',
                        choices=['3d_rgb_blocks', '3d_rgb_multiresolution'],
                        help='color descriptor used')

    parser.add_argument('--texture_descriptor', type=str, default='dct_blocks',
                        choices=['dct_blocks', 'dct_multiresolution', 'lbp_blocks', 'lbp_multiresolution',
                                 'hog_blocks', 'hog_multiresolution', 'wavelet_blocks', 'wavelet_multiresolution'],
                        help='texture descriptor used')

    parser.add_argument('--color_weight', type=float, default=0.33,
                        help='weight for the color descriptor')

    parser.add_argument('--texture_weight', type=float, default=0.33,
                        help='weight for the texture descriptor')

    parser.add_argument('--text_weight', type=float, default=0.0,
                        help='weight for the text descriptor')

    parser.add_argument('--color_metric', type=str, default='Hellinger',
                        choices=['Hellinger', 'Intersection', 'Chi-Squared', 'Correlation'],
                        help='distance metric to compare images')

    parser.add_argument('--texture_metric', type=str, default='Intersection',
                        choices=['Hellinger', 'Intersection', 'Chi-Squared', 'Correlation'],
                        help='distance metric to compare images')

    parser.add_argument('--text_metric', type=str, default='Levensthein',
                        choices=['Levensthein', 'Hamming', 'Damerau'],
                        help='distance metric to compare images')

    parser.add_argument('--number_blocks', type=int, default=16,
                        help='number of blocks in which the image is divided if using the block-based histograms')

    parser.add_argument('--multiresolution_blocks', type=lambda s: [int(item) for item in s.split(',')], default=[1,4,8,16],
                        help='list of numbers of blocks in which the image is divided if using the multiresolution histograms')

    parser.add_argument('--verbose', action='store_true',
                        help='increase output verbosity: show k similar images per each query image')

    args = parser.parse_args(args)
    return args

def args_to_params(args):
    results_path = os.path.join(args.query_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    params = {
        'lists': None, 'paths': None, 'color': None, 'texture': None, 'text': None, 'remove': None
    }
    params['paths'] = {
        'bbdd': args.bbdd_path,
        'query': args.query_path,
        'results': results_path
    }
    if args.use_color:
        params['color'] = {
            'descriptor': args.color_descriptor,
            'weight': args.color_weight,
            'metric': args.color_metric
        }
    if args.use_texture:
        params['texture'] = {
            'descriptor': args.texture_descriptor,
            'weight': args.texture_weight,
            'metric': args.texture_metric
        }
    if args.use_text:
        params['text'] = {
            'weight': args.text_weight,
            'metric': args.text_metric
        }
    if args.remove_bg or args.remove_text:
        params['remove'] = {
            'bg': args.remove_bg,
            'text': args.remove_text,
            'max_paintings': args.max_paintings
        }
    return params

def lists_to_params(params, bbdd_list, query_list):
    params['lists'] = {
        'bbdd': bbdd_list,
        'query': query_list
    }
    return params

def path_to_list(data_path, extension='jpg'):
    path_list = sorted(glob(os.path.join(data_path,'*.'+extension)))
    if not path_list:
        str = 'No .' + extension + ' files found in directory ' + data_path
        sys.exit(str)
    return path_list

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(pickle_path, pickle_file):
    with open(pickle_path, 'wb') as f:
        return pickle.dump(pickle_file, f)
    #return None

def run():
    args = parse_args()
    params = args_to_params(args)
    print(params)

    k = args.map_k

    # # Path to bbdd and query datasets
    # bbdd_path = args.bbdd_path
    # query_path = args.query_path
    # results_path = os.path.join(query_path, 'results')

    bbdd_list = path_to_list(params['paths'].bbdd, extension='jpg')
    query_list = path_to_list(params['paths'].query, extension='jpg')

    params = lists_to_params(params, bbdd_list, query_list)

    # if args.use_text:
    #     text_list = load_pickle(os.path.join(query_path, 'text_boxes.pkl'))

    paintings_predicted_list = retrieval.get_k_images(params, k=max(k))

    save_pickle(os.path.join(params['paths'].results, 'result.pkl'))

    if not args.test:
        evaluation.evaluate(paintings_predicted_list, params, k, verbose=args.verbose)
