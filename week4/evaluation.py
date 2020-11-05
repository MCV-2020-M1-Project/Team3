
import week4.utils as utils

def output_predicted_paintings(query_list, paintings_predicted_list, paintings_groundtruth_list, k):
    for query_image_path in query_list:
        image_id = utils.get_image_id(query_image_path)

        paintings_predicted_image = paintings_predicted_list[image_id]
        paintings_groundtruth_image = paintings_groundtruth_list[image_id]

        print(f'Image: {query_image_path}')

        for painting_id, painting_groundtruth in enumerate(paintings_groundtruth_image):
            print('-> Painting #{}'.format(painting_id))
            print('    Groundtruth: {}'.format(painting_groundtruth))

            # If we detected the painting
            if len(paintings_predicted_image) > painting_id:
                print(f'        {k} most similar images: {paintings_predicted_image[painting_id]}')
            else:
                print('        Painting not detected!!!')

        print('----------------------')

def evaluate(paintings_predicted_list, params, k_list, verbose=False):
    if params['remove'].bg:
        bg_predicted_list = path_to_list(params['paths'].results, extension='png')
        bg_groundtruth_list = path_to_list(params['paths'].query, extension='png')
        # assert len(bg_groundtruth_list) == len(bg_predicted_list)
        evaluation.bg(bg_predicted_list, bg_groundtruth_list)

    # if params['remove'].text_extract:
    #     text_extract_predicted_list = path_to_list(params['paths'].results, extension='txt')
    #     text_extract_groundtruth_list = path_to_list(params['paths'].query, extension='txt')
    #     # assert len(text_groundtruth_list) == len(text_predicted_list)
    #     evaluation.text_extract(text_extract_predicted_list, text_extract_groundtruth_list)

    if params['remove'].text:
        text_boxes_predicted_list = load_pickle(os.path.join(params['paths'].results, 'text_boxes.pkl'))
        text_boxes_groundtruth_list = load_pickle(os.path.join(params['paths'].query, 'text_boxes.pkl'))
        # assert len(text_boxes_predicted_list) == len(text_boxes_groundtruth_list)
        evaluation.text_boxes(text_boxes_predicted_list, text_boxes_groundtruth_list)

    paintings_groundtruth_list = load_pickle(os.path.join(params['paths'].query, 'gt_corresps.pkl'))

    if verbose:
        output_predicted_paintings(params['lists'].query, paintings_predicted_list, paintings_groundtruth_list, max(k_list))

    for k in k_list:
        map = evaluation.mapk(paintings_predicted_list, paintings_groundtruth_list, k)
        print(f'MAP@{k}: {map}')
