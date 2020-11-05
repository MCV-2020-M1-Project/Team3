import os
import sys
import argparse
import pickle

import week4.retrieval as retrieval
import week4.evaluation as evaluation
import week4.utils as utils

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

    parser.add_argument('--color_descriptor', type=str, default='rgb_3d_blocks',
                        choices=['rgb_3d_blocks', 'rgb_3d_multiresolution'],
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

    parser.add_argument('--color_metric', type=str, default='hellinger',
                        choices=['hellinger', 'intersection', 'chi-Squared', 'correlation'],
                        help='distance metric to compare images')

    parser.add_argument('--texture_metric', type=str, default='intersection',
                        choices=['hellinger', 'intersection', 'chi-Squared', 'correlation'],
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
    if not True in (args.use_color, args.use_texture, args.use_text):
        sys.error('No descriptor method specified')

    return params

def lists_to_params(params, bbdd_list, query_list):
    params['lists'] = {
        'bbdd': bbdd_list,
        'query': query_list
    }
    return params

def run():
    args = parse_args()
    params = args_to_params(args)
    print(params)

    k = args.map_k

    # # Path to bbdd and query datasets
    # bbdd_path = args.bbdd_path
    # query_path = args.query_path
    # results_path = os.path.join(query_path, 'results')

    bbdd_list = utils.path_to_list(params['paths']['bbdd'], extension='jpg')
    query_list = utils.path_to_list(params['paths']['query'], extension='jpg')

    params = lists_to_params(params, bbdd_list, query_list)

    # if args.use_text:
    #     text_list = utils.load_pickle(os.path.join(query_path, 'text_boxes.pkl'))

    paintings_predicted_list = retrieval.get_k_images(params, k=max(k))

    utils.save_pickle(os.path.join(params['paths']['results'], 'result.pkl'), paintings_predicted_list)

    if not args.test:
        evaluation.evaluate(paintings_predicted_list, params, k, verbose=args.verbose)
