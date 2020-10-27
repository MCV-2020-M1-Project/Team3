import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'<path to teseract.exe>' #for windows users
from PIL import Image
import operator
import cv2 as cv
import jellyfish as jel


def get_text(img, text_box):
    tl_x=text_box[0][0]
    tl_y=text_box[0][1]
    br_x=text_box[1][0]
    br_y=text_box[1][1]
    
    roi=img[tl_x:br_x,tl_y:br_y]
    text = tess.image_to_string(roi,lang='cat').rstrip() # removes the end whitespaces
    
    return text

def get_text_distance(text_1,text_2,distance_metric="Levensthein"):
    if distance_metric=="Levensthein":
        distance=jel.levenshtein_distance(text_1,text_2)
        
    elif distance_metric=="Hamming":
        distance=jel.hamming_distance(text_1,text_2)
        
    elif distance_metric=="Damerau":
        distance=jel.damerau_levenshtein_distance(text_1,text_2)
        
    elif distance_metric=="Jaro":
        distance=1/jel.jaro_distance(text_1,text_2)
    else:
        print('Metric doesn\'t exist')
    return distance
        
def get_k_images(painting, text_box, bbdd_texts, k="10", distance_metric="Levensthein"):

    text = get_text(painting, text_box)
    
    distances = {}

    for bbdd_id, bbdd_texts in bbdd_texts.items():
        distances[bbdd_id] = get_text_distance(text, bbdd_texts,distance_metric)

    k_predicted_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=False))[:k]

    return [predicted_image[0] for predicted_image in k_predicted_images]
        


###----TEST AREA----
img = cv.imread("data/qsd1_w2/00000.jpg") 
roi = [0,105,110,574]

text = get_text(img,roi)


gt_text =  'Bodego IV'
gt_text_case =  'bodego iv'
fail_text =  'Bodega VV'


dist_lev1=get_text_distance(gt_text,text)
dist_lev2=get_text_distance(gt_text_case,text)
dist_lev3=get_text_distance(fail_text,text)

dist_ham1=get_text_distance(gt_text,text)
dist_ham2=get_text_distance(gt_text_case,text)
dist_ham3=get_text_distance(fail_text,text)


print('{} has a levenshtein distances of {}, {} and {}. Also a hamming distances of {}, {} and {}'.format(text, dist_lev1,dist_lev2,dist_lev3,dist_ham1,dist_ham2,dist_ham3))
 