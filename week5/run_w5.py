import os
import sys
import argparse
import pickle

import week4.retrieval as retrieval
import week4.evaluation as evaluation
import week4.utils as utils

from week4.run_w4 import parse_args, args_to_params, lists_to_params
from week5 import cluster


def run():
    args = parse_args()
    params = args_to_params(args)
    print(params)

    k = args.map_k

    bbdd_list = utils.path_to_list(params['paths']['bbdd'], extension='jpg')
    query_list = utils.path_to_list(params['paths']['query'], extension='jpg')
    params = lists_to_params(params, bbdd_list, query_list)

    if args.cluster_images:
        print("K:=", k)
        cluster.cluster(bbdd_list, max(k))
    else:
        paintings_predicted_list = retrieval.get_k_images(params, k=max(k))

        utils.save_pickle(os.path.join(params['paths']['results'], 'result.pkl'), paintings_predicted_list)

        if not args.test:
            evaluation.evaluate(params, k, verbose=args.verbose)
