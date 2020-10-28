import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #your path to tesseract for windows users
from PIL import Image
import operator
import cv2 as cv
import jellyfish as jel
import os



def get_bbdd_texts(bbdd_path):
    bbdd_texts = {}
    for image_filename in sorted(os.listdir(bbdd_path)):
        if image_filename.endswith('.txt'):
            image_id = int(image_filename.replace('.txt', '').replace('bbdd_', ''))
            f = open(os.path.join(bbdd_path, image_filename), "r")
            line= f.readline()
            if line.strip():
                bbdd_text = line.lower().replace("(","").replace("'","").replace(")","") #ignore case
            else:
                bbdd_text='empty'
            bbdd_texts[image_id] = bbdd_text

            
    return bbdd_texts


def get_text(img, text_box):
    tl_x=text_box[0][0]
    tl_y=text_box[0][1]
    br_x=text_box[1][0]
    br_y=text_box[1][1]
    
    roi=img[tl_x:br_x,tl_y:br_y]
    text = tess.image_to_string(roi,lang='cat').rstrip().lower() # removes the end whitespaces and lower to ignore case
    
    return text

def get_text_distance(text_1,text_2,distance_metric="Levensthein"):
    if distance_metric=="Levensthein":
        distance=jel.levenshtein_distance(text_1,text_2)
        
    elif distance_metric=="Hamming":
        distance=jel.hamming_distance(text_1,text_2)
        
    elif distance_metric=="Damerau":
        distance=jel.damerau_levenshtein_distance(text_1,text_2)
        
    elif distance_metric=="Jaro":
        distance=int(10/jel.jaro_distance(text_1,text_2))
    else:
        print('Metric doesn\'t exist')
    return distance
        
def get_k_images(painting, text_box, bbdd_texts, k=10, distance_metric="Levensthein"):

    text = get_text(painting, text_box)
    
    distances = {}

    for bbdd_id, bbdd_text in bbdd_texts.items():
        
        if bbdd_text!='empty':
            bbdd_text=bbdd_text.replace("(","").replace("'","").replace(")","")
            distances[bbdd_id] = get_text_distance(text, bbdd_text.split(",")[1].rstrip(),distance_metric) #for week 3 dataset change to index 0 = author
        
        else:
            distances[bbdd_id]=100
            
    k_predicted_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=False))[:k]

    return [predicted_image[0] for predicted_image in k_predicted_images]
        


###----TEST AREA----
img = cv.imread("data/qsd1_w2/00000.jpg") 
roi = [[0,105],[110,574]]
bbdd_path='data/BBDD/'

bbdd_texts=get_bbdd_texts(bbdd_path)

predicted_paintings = get_k_images(img,roi,bbdd_texts)
print(predicted_paintings)

