import scipy.spatial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, datasets

def basic(img):
    img = cv.medianBlur(img,5)
    ret,th1 = cv.threshold(img,50,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding w/ blur (v = 50)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.figure()
    return th1

def otsu(img):
    # global thresholding
    ret1,th1 = cv.threshold(img,50,255,cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, th1,
              img, th2]
              #blur, 0, th3]
    titles = ['Original Noisy Image',#'Histogram',
                'Global Thresholding (v=50)',
              'Original Noisy Image',#'Histogram',
                "Otsu's Thresholding"]
              #'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    '''for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.figure()'''
    return th2
#    for i in range(2):
#        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#        #plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#        #plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
#    plt.show()

def add_drift(img, mean, variance):
    pass

def add_sensor_noise(img, mean, variance):
    gauss = np.random.normal(mean, variance ** 0.5, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)

def smear(img, count=30):
    import random, math
    rows, cols = img.shape
    length = random.randrange(200, rows + cols) # in pixels
    smear_width = 5 # pixels
    x_start = random.randrange(0, cols)
    y_start = random.randrange(0, rows)
    #angle = random.uniform(0, 2 * 3.1415)
    #x_end = int(x_start + length * math.cos(angle))
    #y_end = int(y_start + length * math.sin(angle))
    x_end = random.randrange(0, cols)
    y_end = random.randrange(0, rows)

    props = {}
    props['slope'] = (y_end - y_start) / (x_end - x_start)
    props['intercept'] = y_start - x_start * props['slope']
    props['start'] = (x_start, y_start)
    props['end'] = (x_end, y_end)

    #x_end = int(1/slope * length)
    #y_end = int(slope * length)

    #cv.line(img,(x_start,y_start),(x_end ,y_end),(0,0,0),smear_width)
    steps = np.linspace(0, length, count)
    frames = []
    plt.figure()
    xs = []
    ys = []
    for step in steps:
        frame = img.copy()
        x_pos = int(x_start + step / length * (x_end - x_start))
        y_pos = int(y_start + step / length * (y_end - y_start))
        xs.append(x_pos)
        ys.append(y_pos)
        cv.circle(frame, (x_pos, y_pos), smear_width, (0, 0, 0), -1)
        frames.append(frame)
        #plt.imshow(frame,'gray')
        #plt.figure()
        #print((x_pos, y_pos))
    plt.title("Reference image with superimposed occultation")
    plt.imshow(frame,'gray')
    plt.figure()
    plt.title("Reference image with enlarged occultation")
    plt.ylim(rows, 0)
    plt.xlim(0, cols)
    plt.scatter(xs, ys)

    return frames, props
 
def compute_missing_pixels(ref_img, smear_img, row, col):
    pass

def invert(img):
    cv.bitwise_not(img)

def find_anomolies(before, after):
    anomolies = []
    before_bin = otsu(before)
    after_bin = otsu(after)
    #before_bin = before
    #after_bin = after
    for row in range(before_bin.shape[0]):
        for col in range(before_bin.shape[1]):
            if before_bin[row][col] > after_bin[row][col]:
                anomolies.append([col, row])
    return anomolies or None

def find_stars(frame):
    frame_bin = otsu(frame)
    #frame_bin = frame
    stars = []
    for row in range(frame_bin.shape[0]):
        for col in range(frame_bin.shape[1]):
            if frame_bin[row][col]:
                stars.append([row, col])
    print('Found {} star pixels out of {} pixels in ref img'.format(len(stars), frame.size))
    return stars

    



def find_smears(frames):
    import itertools
    stars = find_stars(frames[0])
    kdtree = scipy.spatial.KDTree(stars)
    all_anomolies = []
    for i in range(1, len(frames)):
        # O(n^2), but doable by ASIC
        anomolies = find_anomolies(frames[i-1], frames[i])
        if not anomolies:
            continue
        all_anomolies += anomolies
        #pairs = kdtree.query_pairs(800)
        print('Found {} anomolies in frame {}'.format(len(anomolies), i))
    all_anomolies = list(filter(None, all_anomolies))
    print('anoms:', all_anomolies)

    x, y = zip(*all_anomolies)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(np.asarray(x).reshape(-1, 1), np.asarray(y).reshape(-1, 1))

    coef = ransac.estimator_.coef_
    line_info = {}
    [line_info['slope']] = ransac.estimator_.coef_
    [line_info['intercept']] = ransac.estimator_.intercept_


    plt.figure()
    plt.title('Anomolies')
    plt.xlim(0, frames[0].shape[1])
    plt.ylim(frames[0].shape[0], 0)
    plt.scatter(x, y)
        #cv.circle(frames[0], (anom[0], anom[1]), 10, (0, 0, 0), -1)
    return line_info
    """
    if len(all_anomolies) < 2:
        Exception('Not enough anomolies')
    pairs = list(itertools.combinations(all_anomolies, 2))
    print('anoms:', all_anomolies)
    #print('pairs:', list(pairs))
    lines = []
    for pair in pairs:
        #x, y = zip(*pair)
        #print('pair:', pair)
        x = (pair[0][0], pair[1][0])
        y = (pair[0][1], pair[1][1])
        #print('[x,y]:', [x, y])
        #coefs.append(np.polyfit(x, y, 1))
        dx = x[1] - x[0]#pairs[1][0] - pairs[0][0]
        dy = y[1] - y[0]#pairs[1][1] - pairs[0][1]
        try:
            slope = dy / dx
        except ZeroDivisionError:
            continue
            #slope = float('Inf')
        # y = mx + b
        # b = y - mx
        #intercept = y[0] - x[0] * slope
        #coefs.append((slope, intercept))
        #filtered_pairs.append(pair)

        line = {}
        line['intercept'] = y[0] - x[0] * slope
        line['slope'] = slope
        line['points'] = pair
        lines.append(line)

    #print('coefs:', coefs)


    # Now that we have linear eq, compute mean squared error
    # line with the lowest error wins
    x_true, y_true = zip(*all_anomolies)
    for line in lines:
        y_pred = [line['slope'] * x + line['intercept'] for x in x_true] # y=mx+b
        line['error'] = mean_squared_error(y_true, y_pred)

    sorted_lines = sorted(lines, key=lambda x: x['error'])
    print("Best lines: {}".format(sorted_lines[0:5]))

    # Now that we have a starting point, discard outliers and try again
    #remove_outliers()
    return sorted_lines[0]
    """
#def remove_

def optical_flow(ref_img, smear_img):
    #star_pos = [(row, col) for row, col in (range(ref_img.shape[0]), range(ref_img.shape[1]))]
    star_map = []
    for row in range(ref_img.shape[0]):
        for col in range(ref_img.shape[1]):
            if ref_img[row][col]:
                star_map.append((row, col))
    star_map = np.ndarray(star_map)
    
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv.calcOpticalFlowPyrLK(ref_img, smear_img, star_map, None, **lk_params)


def graph_props(img, actual, estimate):
    cv.line(img,actual['start'],actual['end'],(255,0,0),5)

    x_start, x_end = (actual['start'][0], actual['end'][0])
    y_start = int(estimate['slope'] * x_start + estimate['intercept'])
    y_end = int(estimate['slope'] * x_end + estimate['intercept'])

    cv.line(img,(x_start, y_start), (x_end, y_end), (127,0,0),5)
    
    plt.figure()
    plt.title("Graph props")
    plt.imshow(img, 'gray')


base_img = cv.imread('pics/night_sky0.jpg',0)
frames, props = smear(base_img)
line_props = find_smears([base_img] + frames)
print('estimated props', line_props)
print('actual props:', props)
graph_props(base_img, props, line_props)
#global_img = basic(img)
#otsu_img = otsu(img)
#find_smear(base_img, smear_img)
#optical_flow(base_img, smear_img)
plt.show()
