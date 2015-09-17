from datetime import datetime

import numpy as np
import cv2
import Rect_functions

# debug constant TODO: fix this thing use __debug__ instead
debug = False

# DEFINICION DE VARIABLES GLOBALES
refPt = []  # ARRAY DE PTOS DE REFERENCIA
selected = False  # VARIABLE PARA SABER SI ESTA SELECCIONANDO
following = False  # VARIABLE PARA SABER SI ESTA SIGUIENDO

# VARIABLES PARA EL CAMSHIFT
track_window = ()

# setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 1000)

roi_hist = ()
roi = ()
selection = ()
frame = ()
percent = 0.3


def click_on_mouse(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, selected, following, track_window, term_criteria, roi_hist, roi

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that seleccionando is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [x]
        refPt.append(y)
        selected = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the selection operation is finished
        refPt.append(x)
        refPt.append(y)
        selected = False
        following = True

        # setup initial location of window
        w = refPt[2] - refPt[0]
        h = refPt[3] - refPt[1]
        r = refPt[1]
        c = refPt[0]
        track_window = (c, r, w, h)

        # set up the ROI for tracking
        roi = selection[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_m = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255, 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask_m, [255], [0, 255])

        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(roi_hist)

        # after getting the max color location the histogram is reset to 0
        roi_hist = 0*roi_hist

        # treshold to determine color variation sensibility
        threshold = 4

        # setting 1 around the max location creates the color search mask
        for i in range(0, threshold, 1):
            roi_hist[max_loc[1] + i] = 1
            roi_hist[max_loc[1] - i] = 1

        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        cv2.imshow("ROI", roi)
        cv2.imwrite("ROI.jpg", frame[r:r + h, c:c + w])

    # draw a rectangle around the region of interest
    if selected:
        cv2.rectangle(selection, (refPt[0], refPt[1]), (x, y), (0, 255, 0), 2)
        cv2.imshow("ROI", selection)


def buscar_objeto():  # Todo: clean up everything

    global selected, following, track_window, term_criteria, roi_hist, roi

    print("Searching")

    # load the new frame
    background = frame.copy()

    _hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _mask = cv2.inRange(_hsv, np.array((0, 80., 80.)), np.array((255, 255., 255.)))
    _dst = cv2.calcBackProject([_hsv], [0], roi_hist, [0, 255], 1)
    _dst = _mask * _dst
    _kernel = np.ones((2, 2), np.uint8)

    _dst = cv2.erode(_dst, _kernel, iterations=3)
    # do a lot of iterations to eliminate tiny contours
    _dst = cv2.dilate(_dst, _kernel, iterations=20)  # Todo: magic number why 20?

    imi, contours, hier = cv2.findContours(_dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    list_rec = []  # lists the rectangles that surround a contour

    for cnt in contours:

        # for each contour define a rectangle that contains it
        # avoid small contours (camera glitches or noise)
        if 100 < cv2.contourArea(cnt):  # Todo: another magic number
            r = cv2.boundingRect(cnt)
            # cv2.drawContours(fondo,[cnt],0,(0,255,0),2)
            list_rec.append(r)

    Rect_functions.rect_grouper(list_rec)

    # for r2 in listrec:
    #     cv2.rectangle(fondo,(r2[0],r2[1]),(r2[0]+r2[2],r2[1]+r2[3]),(255,0,0),2,8,0)
    #     cv2.imwrite("imgo.jpg",fondo)

    background = cv2.cvtColor(background, cv2.CV_8U)
    (background_height, background_width) = background.shape[:2]

    # cargo el template
    template = cv2.imread("template.jpg")
    template = cv2.cvtColor(template, cv2.CV_8U)
    (template_height_0, template_width_0) = template.shape[:2]

    found = None
    for r in list_rec:

        roi_offset_x = r[0]
        roi_offset_y = r[1]
        
        # get each region of interest
        roi = background[r[1]:(r[1] + r[3]), r[0]:(r[0] + r[2])]
        (roi_height, roi_width) = roi.shape[:2]

        cv2.imwrite("imgo.jpg", roi)

        r_h = float(roi_height) / float(template_height_0)
        r_w = float(roi_width) / float(template_width_0)

        # check which dimension has de smaller proportion and use that to scale the template
        if r_h > r_w:
            first_scale = float(roi_width) / float(template_width_0)
        else:
            first_scale = float(roi_height) / float(template_height_0)

        # discard iteration if the scaling factor is too small
        if first_scale <= 0.1:
            # print "continue"
            continue

        scaled_template = cv2.resize(template, (0, 0), fx=first_scale, fy=first_scale)

        # check from different template sizes the one who adjust better to the template

        for scale in np.linspace(0.1, 1.0, 10)[::-1]:  # resize from 0.1 to 1 template

            if scale * first_scale < 0.1: # if the template scale is lower than the 10%, discard that contour
                continue

            scaling_template = cv2.resize(scaled_template, (0, 0), fx=scale, fy=scale)

            (scaling_template_height, scaling_template_width) = scaling_template.shape[:2]

            # r = scaled_roi.shape[1] / scaled_template.shape[1]

            if scaling_template_height >= roi_height or scaling_template_width >= roi_width:
                print("Dimension error, scaled template is bigger than the roi region")
                break

            # TODO: esto no deberia ser contra el roi no contra el fondo?
            # if resized.shape[0] >= background_height or resized.shape[1] >= background_width:
            #     print("Error en dimensiones")
            #     break

            # do the template matching
            result = cv2.matchTemplate(roi, scaling_template, cv2.TM_CCOEFF_NORMED)

            (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)

            # if False:  # poner en True en caso de querer ver paso a paso la deteccion del objeto y valores de matching
            #     # draw a bounding box around the detected region
            #     clone = roi.copy()
            #     cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + scaling_template_width, maxLoc[1] + scaling_template_height), (0, 0, 255), 2)
            #     cv2.imshow("Visualize", clone)
            #     print maxVal
            #     cv2.waitKey(0)

            if found is None or max_val > found[0]:
                found = (max_val, (max_loc[0] + roi_offset_x, max_loc[1] + roi_offset_y),
                         scaling_template_height, scaling_template_width)

    (max_val, max_loc, h, w) = found

    if max_val > 0.4:  # umbral para certeza de deteccion de objeto #TODO: fix threshold
        following = True
        # setup initial location of window
        w = int(w)
        h = int(h)
        r = int(max_loc[1])  # y
        c = int(max_loc[0])  # x
        track_window = (c, r, w, h)
        clone = background.copy()
        cv2.rectangle(clone, (c, r), (c + w, r + h), (0, 0, 255), 2)
        cv2.imshow("Object found", clone)
    else:
        print "Object not detected"


cv2.namedWindow("TiempoReal")
cv2.setMouseCallback("TiempoReal", click_on_mouse)

cap = cv2.VideoCapture(0)

# take first frame of the video
ret, frame = cap.read()

while 1:
    time = datetime.now()
    ret, frame = cap.read()

    if ret == True:
        selection = frame.copy()
        if following == True:

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0, 80., 80.)), np.array((255, 255., 255.)))
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 255], 1)
            dst = mask * dst
            kernel = np.ones((2, 2), np.uint8)

            dst = cv2.erode(dst, kernel, iterations=3)
            dst = cv2.dilate(dst, kernel, iterations=9)
            cv2.imshow('TOTAL', 255 * dst)


            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_criteria)
            # cv2.imshow("BackProjection", dst)
            if (track_window[2] > 3 * track_window[3]) or (
                        track_window[3] > 3 * track_window[2]):  # Limites de desision para dejar de seguir
                following = False

            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('TiempoReal', img2)
        else:
            cv2.imshow('TiempoReal', frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.imwrite("prueba.jpg", 255 * dst)
            break
        elif k == 111:
            buscar_objeto()
        elif k == 112:
            following = False

    else:
        break
    time = datetime.now() - time
    # print round(1/time.total_seconds())

cv2.destroyAllWindows()
cap.release()
