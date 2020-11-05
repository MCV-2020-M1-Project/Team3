import os
import sys
import pickle
from glob import glob

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

def get_image_id(image_path):
    image_filename = image_path.split('/')[-1]
    image_id = image_filename.split('.')[0]

    return image_id

def zigzag(input, n_coefs):
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]

    output = []

    while (v < vmax) and (h < hmax) and (len(output) < n_coefs):
        output.append(input[v,h])

        if ((h + v) % 2) == 0:                 # going up

            if (v == vmin):

                if (h == hmax-1):
                    v = v + 1

                else:
                    h = h + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                v = v + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                v = v - 1
                h = h + 1

        else:                                    # going down
            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                h = h + 1

            elif (h == hmin):                  # if we got to the first column
                if (v == vmax -1):
                    h = h + 1

                else:
                    v = v + 1

            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                v = v + 1
                h = h - 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output.append(input[v, h])
            break

    return output

def inverse_zigzag(input, vmax, hmax):

    h = 0
    v = 0

    hmin = 0
    vmin = 0

    output = np.zeros((vmax, hmax))

    i = 0

    while True:
        if ((h + v) % 2) == 0:                 # going up
            if (v == vmin):
                output[v, h] = input[i]        # if we got to the first line

                if (h == hmax-1):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1

        else:                                    # going down
            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif (h == hmin):                  # if we got to the first column
                output[v, h] = input[i]
                if (v == vmax-1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1

            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output[v, h] = input[i]
            break

    return output
