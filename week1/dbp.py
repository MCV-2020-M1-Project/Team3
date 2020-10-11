import os
import operator
import cv2 as cv

OPENCV_COLOR_SPACES = {
    "RGB": cv.COLOR_BGR2RGB,
    "GRAY": cv.COLOR_BGR2GRAY,
    "LAB": cv.COLOR_BGR2LAB,
    "YCrCb": cv.COLOR_BGR2YCrCb,
    "HSV" : cv.COLOR_BGR2HSV
}

OPENCV_DISTANCE_METRICS = {
    "Correlation": cv.HISTCMP_CORREL,
    "Chi-Squared": cv.HISTCMP_CHISQR,
    "Intersection": cv.HISTCMP_INTERSECT,
    "Hellinger": cv.HISTCMP_BHATTACHARYYA
}

class dbp:

    def compute_histogram(self, image_path, color_space="RGB"):
        """
        compute_histogram()

        Function to compute ...
        """

        hist_channels = [0]
        hist_bins = [256]
        hist_range = [0, 256]

        # if not gray --> 3 channels (RGB, HSV, ...)
        if color_space != "GRAY":
            hist_channels = hist_channels + [1, 2]
            hist_bins = hist_bins * 3  # hist_bins = [256, 256, 256]
            hist_range = hist_range * 3  # hist_range = [0, 256, 0, 256, 0, 256]

        img = cv.imread(image_path)
        hist = cv.calcHist([cv.cvtColor(img, OPENCV_COLOR_SPACES[color_space])], hist_channels, None, hist_bins,
                           hist_range)
        hist = cv.normalize(hist, hist, alpha=0, beta=1,
                            norm_type=cv.NORM_MINMAX).flatten()  # change histogram range from [0,256] to [0,1]

        return hist

    def process_images_as_histograms(self, path, color_space="RGB"):
        image_set = sorted(os.listdir(path))
        histograms = {}

        for filename in image_set:
            if filename.endswith('.jpg'):
                number = int(filename.replace('.jpg', '').replace('bbdd_', ''))
                histogram = self.compute_histogram(os.path.join(path, filename), color_space)
                histograms[number] = histogram

        return histograms

    def find_top_similar(self, image_name, folder_path, bbdd_histograms, k="10", distance_metric="Hellinger", color_space="RGB"):
        results = {}
        reverse = False

        if distance_metric in ("Correlation", "Intersection"):
            reverse = True

        if image_name.endswith('.jpg'):
            number = int(image_name.replace('.jpg', ''))
            hist = dbp.compute_histogram(os.path.join(folder_path, image_name), color_space)

            distances = {}
            for bbdd in bbdd_histograms.keys():
                distances[bbdd] = cv.compareHist(hist, bbdd_histograms[bbdd], OPENCV_DISTANCE_METRICS[distance_metric])

            results[number] = (sorted(distances.items(), key=operator.itemgetter(1), reverse=reverse))[:k]

        return results
