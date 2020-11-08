import cv2 as cv
import numpy as np
from tqdm import tqdm

import sys

import multiprocessing.dummy as mp
from functools import partial
from itertools import repeat

import week4.masks as masks
import week4.histograms as histograms
import week4.text_boxes as tb
import week4.noise_removal as noise
import week4.utils as utils
import week4.feature_descriptors as fd


def image_to_paintings(image_path, params):
    img = cv.imread(image_path)
    image_id = utils.get_image_id(image_path)

    paintings = [img]

    text_boxes = [None]
    if params['remove'] is not None:
        if params['remove']['bg']:
            [paintings, paintings_coords] = masks.remove_bg(img, params, image_id)

        if params['remove']['noise']:
            paintings = noise.denoise_paintings(paintings, params, image_id)

        if params['remove']['text']:
            [paintings, text_boxes] = tb.remove_text(paintings, paintings_coords)

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


def get_matches(bbdd_surf, query_surf):

    query_matches_image = []
    for q in query_surf:
        query_matches_paintings=[]
        q_kp, q_des = q
        if len(q_kp) > 0:
            for bbdd in bbdd_surf:
                kp, des = bbdd
                if len(kp) > 0:
                    matches = fd.match_descriptors(q_des, des)
                    if len(matches) > 2:
                        query_matches_paintings.append([bbdd_surf.index(bbdd), matches])
        query_matches_image.append(query_matches_paintings)

    return query_matches_image

def sort(q_matches):
    def calculate_distance(q_match):
        dist = 0
        for m in q_match:
            dist += m.distance
        return dist / len(q_match)

    result = []
    for item in q_matches:
        item.sort(key=lambda x: calculate_distance(item[1]))
        if len(item[1]) < 10:
            result.append(-1)
        else:
            result.append(item[0])

    return result

def get_top_matches(params, k=5, threshold=400):

    all_matches = []
    processes = 4
    paintings_predicted_list = []
    with mp.Pool(processes=processes) as p:
        image_to_paintings_partial = partial(image_to_paintings, params=params)

        print('---Extracting paintings from images (optional: removing background or text)---')
        [paintings, text_boxes] = zip(*list(tqdm(p.imap(image_to_paintings_partial,
                                                        [path for path in params['lists']['query']]),
                                                 total=len(params['lists']['query']))))
        matched = []

        compute_bbdd_surf_partial = partial(fd.surf_descriptor,
                                            threshold=threshold)
        print('---Computing bbdd_surf---')
        bbdd_surf = list(tqdm(p.imap(compute_bbdd_surf_partial,
                                     [path for path in params['lists']['bbdd']])))

        compute_query_surf_partial = partial(fd.surf_descriptor_painting,
                                            threshold=threshold)
        print('---Computing query_surf---')
        query_surf = list(tqdm(p.imap(compute_query_surf_partial,
                                      [p for p in paintings])))

        compute_matches_partial = partial(get_matches,
                                         bbdd_surf)
        print('---Computing matches---')
        matched.append(list(tqdm(p.imap(compute_matches_partial,
                         [query for query in query_surf]))))

        for im in range(len(matched)):
            qlist = []
            for im_m in matched[im]:
                im_list = []
                for match in im_m:
                    im_list.append(sort(match[1]))
                qlist.append(im_list[:k])
            paintings_predicted_list.append(qlist)
    print(paintings_predicted_list)
    return paintings_predicted_list


def get_top_matches_sift(params, k=5, threshold=400):
    
    all_matches = []
    processes = 4
    paintings_predicted_list = []
    with mp.Pool(processes=processes) as p:
        image_to_paintings_partial = partial(image_to_paintings, params=params)

        print('---Extracting paintings from images (optional: removing background or text)---')
        [paintings, text_boxes] = zip(*list(tqdm(p.imap(image_to_paintings_partial,
                                                        [path for path in params['lists']['query']]),
                                                 total=len(params['lists']['query']))))
        all_distances = []

        compute_bbdd_sift_partial = partial(fd.sift_descriptor,
                                            threshold=threshold)
        print('---Computing bbdd_sift---')
        bbdd_sift = list(tqdm(p.imap(compute_bbdd_sift_partial,
                                     [path for path in params['lists']['bbdd']])))

        compute_query_sift_partial = partial(fd.sift_descriptor,
                                            threshold=threshold)
        print('---Computing query_sift---')
        query_sift = list(tqdm(p.imap(compute_query_sift_partial,
                                      [path for path in params['lists']['query']])))

        compute_matches_partial = partial(get_matches_sift,
                                         bbdd_sift, k=k)
        print('---Computing matches---')
        all_distances.append(list(tqdm(p.imap(compute_matches_partial,
                         [query for query in query_sift]))))

        for q in range(len(paintings)):
            qlist = []
            for sq in range(len(paintings[q])):
                dist = np.array(all_distances[0][q][sq])[1]

                for idx, f in range(1, len(all_distances)):
                    dist += all_distances[f][q][sq]

                dist = np.array([x.distance for x in dist])

                nearest_indices = np.argsort(dist)[:k]
                result_list = [index for index in nearest_indices]
                qlist.append(result_list)
            paintings_predicted_list.append(qlist)

    print(paintings_predicted_list)
    return paintings_predicted_list

def get_matches_sift(bbdd_sift, query_sift, k=5):
    
    query_matches = []

    q_kp, q_des = query_sift
    if len(q_kp) > 0:
        for bbdd in bbdd_sift:
            kp, des = bbdd
            if len(kp) > 0:
                matches = fd.match_descriptors_sift(q_des, des)

                good_matches = []
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        good_matches.append(m)

                if len(good_matches) > 2:
                    query_matches.append([bbdd_sift.index(bbdd), good_matches])

    def calculate_distance(q_match):
        dist = 0
        for m in q_match:
            dist += m.distance
        return dist / len(q_match)

    result = []
    for item in query_matches:
        idx, match = item
        item.sort(key=lambda x: calculate_distance(match))
        if len(item) == 0:
            result.append([-1])
        else:
            result.append(item)

    return result