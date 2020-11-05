def get_image_id(image_path):
    image_filename = image_path.split('/')[-1]
    image_id = int(image_filename.split('.')[0])

    return image_id
