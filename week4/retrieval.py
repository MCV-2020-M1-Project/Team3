import cv2 as cv

from itertools import repeat
import multiprocessing as mp

import week4.masks as masks
import week4.text_boxes as text_boxes

def get_k_images(painting, method_texture, bbdd_histograms, text_box, method="M1", k="10", n_bins=8, distance_metric="Hellinger", color_space="RGB", block_size=16):

    reverse = True if distance_metric in ("Correlation", "Intersection") else False

    if method == "M1":
        hist = compute_histogram_blocks_texture(painting, method_texture, text_box, n_bins, color_space, block_size)
    else:
        hist = compute_multiresolution_histograms_texture(painting, method_texture, text_box, n_bins, color_space)

    distances = {}

    for bbdd_id, bbdd_hist in bbdd_histograms.items():
        distances[bbdd_id] = cv.compareHist(hist, bbdd_hist, OPENCV_DISTANCE_METRICS[distance_metric])
        # distances[bbdd_id] = metrics.chi2_distance(hist, bbdd_hist)

    k_predicted_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=reverse))[:k]

    return [predicted_image[0] for predicted_image in k_predicted_images], distances

def image_to_paintings(image_path, params):
    img = cv.imread(image_path)
    image_id = utils.get_image_id(image_path)

    if params['remove'].bg:
        [paintings, paintings_coords] = masks.remove_bg(img, params, image_id)

    text_boxes = None
    if params['remove'].text:
        [paintings, text_boxes] = text_boxes.remove_text(paintings, paintings_coords)

    return [paintings, text_boxes]

def get_k_images(params, k):

    pool_processes = 4

    paintings_predicted_list = []

    print('...Extracting paintings from images (optional: removing background or text)...')
    with mp.Pool(processes=pool_processes) as p:

        image_to_paintings_partial = partial(image_to_paintings, params=params)

        [paintings, text_boxes] = list(tqdm(p.imap(image_to_paintings_partial,
                                                  [path for path in params['lists'].query]),
                                                  total=len(params['lists'].query)))
        print('Done!')

        all_results = []
        weights = []
        if params["color"] is not None:
            print('...Computing color histograms and distances...')

            extract_features_func = partial(extract_features, descriptor=params["color"]["descriptor"],bins=params["color"]["bins"])
            color_distance_func = partial(compute_distance, metric=params["color"]["metric"])

            # descriptors extraction
            query_descriptors = p.map(lambda query: [extract_features_func(img, mask=m) for (img, m) in query], queries)
            image_descriptors = p.map(lambda path: extract_features_func(path2img(path)), museum_list)

            # comparison against database. Score is weighted with the value from params.
            results = [[p.starmap(lambda q, db: params["color"]["weight"] * color_distance_func(q, db),
                                 [(query_desc, db_desc) for db_desc in image_descriptors])
                       for query_desc in query_descs]
                       for query_descs in query_descriptors]

            all_results.append(results)

        if params["texture"] is not None:
            print('texture being used')
            extract_features_func = partial(extract_textures, descriptor=params["texture"]["descriptor"],bins=params["texture"]["bins"])
            color_distance_func = partial(compute_distance, metric=params["texture"]["metric"])
            # descriptors extraction
            query_descriptors = p.map(lambda query: [extract_features_func(img, mask=m) for (img, m) in query], queries)
            image_descriptors = p.map(lambda path: extract_features_func(path2img(path)), museum_list)

            # comparison against database. Score is weighted with the value from params.
            results = [[p.starmap(lambda q, db: params["texture"]["weight"] * color_distance_func(q, db),
                                 [(query_desc, db_desc) for db_desc in image_descriptors])
                       for query_desc in query_descs]
                       for query_descs in query_descriptors]

            all_results.append(results)

        if params["text"] is not None:
            print('text being used')
            text_distance_func = partial(compare_texts, similarity=params["text"]["metric"])
            # descriptors extraction
            query_descriptors = p.starmap(extract_txt, zip(queries, text_list))
            image_descriptors = p.map(read_GT_txt, museum_list)
            # comparison against database. Score is weighted with the value from params.
            results = [[p.starmap(lambda q, db: params["text"]["weight"] * (text_distance_func(q, db)),
                                 [(query_desc, db_desc) for db_desc in image_descriptors])
                       for query_desc in query_descs]
                       for query_descs in query_descriptors]

            all_results.append(results)

        if len(all_results) == 0:
            print("[ERROR] You did not specify any feature extraction method.")
            return None

        # we sum the color/text/textures scores for each query and retrieve the best ones
        dist = np.sum(np.array(all_results), axis=0)
        for q in range(len(queries)):
            qlist = []
            for sq in range(len(queries[q])):
                dist = np.array(all_results[0][q][sq])
                for f in range(1, len(all_results)):
                    dist += all_results[f][q][sq]
                nearest_indices = np.argsort(dist)[:k]
                result_list = [index for index in nearest_indices]
                qlist.append(result_list)
            paintings_predicted_list.append(qlist)

    return paintings_predicted_list
