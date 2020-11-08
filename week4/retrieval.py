import cv2 as cv
import numpy as np
from tqdm import tqdm

import sys

import multiprocessing.dummy as mp
from functools import partial
from itertools import repeat

import week4.masks as masks
import week4.histograms as histograms
import week4.text_boxes as text_boxes
import week4.noise_removal as noise
import week4.utils as utils
import week4.feature_descriptors as fd

def image_to_paintings(image_path, params):
    img = cv.imread(image_path)
    image_id = utils.get_image_id(image_path)

    paintings=[img]

    text_boxes=[None]
    if params['remove'] is not None:
        if params['remove']['bg']:
            [paintings, paintings_coords] = masks.remove_bg(img, params, image_id)

        if params['remove']['noise']:
            paintings = noise.denoise_paintings(paintings, params, image_id)

        if params['remove']['text']:
            [paintings, text_boxes] = text_boxes.remove_text(paintings, paintings_coords)

    return [paintings, text_boxes]

def get_k_images(params, k):

    pool_processes = 4

    paintings_predicted_list = []

    with mp.Pool(processes=pool_processes) as p:

        image_to_paintings_partial = partial(image_to_paintings, params=params)

        print('---Extracting paintings from images (optional: removing background or text)---')
        [paintings, text_boxes] = zip(*list(tqdm(p.imap(image_to_paintings_partial,
                                                  [path for path in params['lists']['query']]),
                                                  total=len(params['lists']['query']))))
        all_distances = []

        if params['color'] is not None:
            compute_bbdd_histograms_partial = partial(histograms.compute_bbdd_histograms,
                                                      descriptor=params['color']['descriptor'])

            print('---Computing color bbdd_histograms---')
            bbdd_histograms = list(tqdm(p.imap(compute_bbdd_histograms_partial,
                                              [path for path in params['lists']['bbdd']]),
                                              total=len(params['lists']['bbdd'])))

            print('---Computing color query_histograms and distances---')
            color_distances = histograms.compute_distances(paintings, text_boxes, bbdd_histograms,
                                                           descriptor=params['color']['descriptor'],
                                                           metric=params['color']['metric'],
                                                           weight=params['color']['weight'])
            all_distances.append(color_distances)

        if params['texture'] is not None:
            compute_bbdd_histograms_partial = partial(histograms.compute_bbdd_histograms,
                                                      descriptor=params['texture']['descriptor'])

            print('...Computing texture bbdd_histograms...')
            bbdd_histograms = list(tqdm(p.imap(compute_bbdd_histograms_partial,
                                              [path for path in params['lists']['bbdd']]),
                                              total=len(params['lists']['bbdd'])))

            print('---Computing texture query_histograms and distances---')
            texture_distances = histograms.compute_distances(paintings, text_boxes, bbdd_histograms,
                                                           descriptor=params['texture']['descriptor'],
                                                           metric=params['texture']['metric'],
                                                           weight=params['texture']['weight'])

            all_distances.append(texture_distances)

        # if params['text'] is not None:
        #     print('...Computing text histograms and distances...')
        #
        #     compute_bbdd_histograms_partial = partial(histograms.compute_bbdd_histograms,
        #                                               descriptor=params['text'].text_descriptor,
        #                                               params=params)
        #
        #     bbdd_histograms = list(tqdm(p.imap(compute_bbdd_histograms_partial,
        #                                       [path for path in params['lists'].bbdd]),
        #                                       total=len(params['lists'].bbdd)))
        #
        #     text_distances = histograms.compute_distances(paintings, text_boxes, bbdd_histograms,
        #                                                    descriptor=params['text'].text_descriptor,
        #                                                    metric=params['text'].metric,
        #                                                    weight=params.['text'].weight)
        #
        #
        #     all_distances.append(text_distances)

        # dist = np.sum(np.array(all_results), axis=0)
        for q in range(len(paintings)):
            qlist = []
            for sq in range(len(paintings[q])):
                dist = np.array(all_distances[0][q][sq])
                for f in range(1, len(all_distances)):
                    dist += all_distances[f][q][sq]
                nearest_indices = np.argsort(dist)[:k]
                result_list = [index for index in nearest_indices]
                qlist.append(result_list)
            paintings_predicted_list.append(qlist)

    return paintings_predicted_list



def get_matches_orb(bbdd_path,query_list,params,k):
    def calculate_distance(matches):
        dist = 0
        for m in matches:
            dist += m.distance
        return dist / len(matches)

    bbdd_list = utils.path_to_list(bbdd_path, extension='jpg')
    bbdd_descriptors = fd.compute_bbdd_orb_descriptors(bbdd_list)

    pool_processes = 4

    with mp.Pool(processes=pool_processes) as p:
        image_to_paintings_partial = partial(image_to_paintings, params=params)

        print('---Extracting paintings from images (optional: removing background or text)---')
        [paintings, text_boxes] = zip(*list(tqdm(p.imap(image_to_paintings_partial,
                                                        [path for path in params['lists']['query']]),
                                                 total=len(params['lists']['query']))))

    query_matches=[]
    for painting in paintings:
        for im in painting:
            kp, des = fd.orb_descriptor(im)
            matching = []
            if len(kp) > 0:
                query_painting = []
                for bbddkpdes in bbdd_descriptors:
                    bd_kp, bd_des = bbddkpdes
                    if len(bd_kp) > 0:
                        matches = fd.match_descriptors(des, bd_des)
                        if len(matches) > 2:
                            matching.append([bbdd_descriptors.index(bbddkpdes), matches])
                    matching.sort(key=lambda x: calculate_distance(x[1]))
                if len(matching) == 0:
                    query_matches.append(-1)
                else:
                    query_matches.append(matching[:k])
        #query_matches.append(query_painting)

    return query_matches
    #all_distances = fd.compute_bbdd_orb_query_descriptors(params['orb']['query'], bbdd_descriptors)


def get_top_matches(query_list, bbdd_list, k = 5, threshold = 400):
    def calculate_distance(matches):
        dist = 0
        for m in matches:
            dist += m.distance
        return dist/len(matches)


    bbdd_surf = []
    for bbdd_filename in bbdd_list:
        im = cv.imread(bbdd_filename)
        bbdd_surf.append(fd.orb_descriptor(im,False))

    query_matches = []
    for query in query_list:
        im = cv.imread(query)
        kp, des = fd.orb_descriptor(im)
        matching = []
        if len(kp) > 0:
            for bbddkpdes in bbdd_surf:
                bd_kp, bd_des = bbddkpdes
                if len(bd_kp) > 0:
                    matches = fd.match_descriptors(des, bd_des)
                    if len(matches) > 2:
                        matching.append([bbdd_surf.index(bbddkpdes), matches])
                matching.sort(key=lambda x: calculate_distance(x[1]))
            if len(matching) == 0:
                query_matches.append(-1)
            else:
                query_matches.append(matching[:k])
    return query_matches

