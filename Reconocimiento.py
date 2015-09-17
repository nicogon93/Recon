import numpy as np
import cv2
import Rect_functions
from datetime import datetime

#DEFINICION DE VARIABLES GLOBALES
refPt = [] #ARRAY DE PTOS DE REFERENCIA
seleccionando = False #VARIABLE PARA SABER SI ESTA SELECCIONANDO
siguiendo=False #VARIABLE PARA SABER SI ESTA SIGUIENDO

#VARIABLES PARA EL CAMSHIFT
track_window=()
term_crit=()
roi_hist=()
roi=()
seleccion=()
frame=()
porcentaje=0.3


def click_on_mouse(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, seleccionando,siguiendo,track_window,term_crit,roi_hist,roi

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that seleccionando is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [x]
        refPt.append(y)
        seleccionando = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the seleccionando operation is finished
        refPt.append(x)
        refPt.append(y)
        seleccionando = False
        siguiendo = True
        # setup initial location of window
        w=refPt[2]-refPt[0]
        h=refPt[3]-refPt[1]
        r=refPt[1]
        c=refPt[0]
        track_window = (c,r,w,h)

        # set up the ROI for tracking
        roi = seleccion[r:r+h, c:c+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0.,0.,0.)), np.array((255,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[255],[0,255])
        (minVal, maxVal,minLoc, maxLoc) = cv2.minMaxLoc(roi_hist)
        aux=0*roi_hist
        aux[maxLoc[1]]=1
        aux[maxLoc[1]+1]=1
        aux[maxLoc[1]-1]=1
        aux[maxLoc[1]+2]=1
        aux[maxLoc[1]-2]=1
        aux[maxLoc[1]+3]=1
        aux[maxLoc[1]-3]=1
        aux[maxLoc[1]+4]=1
        aux[maxLoc[1]-4]=1
        roi_hist = aux+0*roi_hist
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 1000 )
        cv2.imshow("ROI", roi)
        #cv2.imwrite("template.jpg",frame[r:r+h, c:c+w])

    # draw a rectangle around the region of interest
    if seleccionando == True:
        cv2.rectangle(seleccion, (refPt[0],refPt[1]),(x,y), (0, 255, 0), 2)
        cv2.imshow("ROI", seleccion)


def buscar_objeto():
    print("Buscando")
    global seleccionando,siguiendo,track_window,term_crit,roi_hist,roi

    #cargo la imagen a analizar
    fondo = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0,80.,80.)), np.array((255,255.,255.)))
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,255],1)
    dst=mask*dst
    kernel = np.ones((2,2),np.uint8)

    dst = cv2.erode(dst,kernel,iterations = 3)
    dst = cv2.dilate(dst,kernel,iterations = 20)

    imi,contours, hier = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    listrec=[]
    for cnt in contours:
        if 100<cv2.contourArea(cnt):
            #cv2.drawContours(fondo,[cnt],0,(0,255,0),2)
            r=cv2.boundingRect(cnt)
            listrec.append(r)
    #print listrec
    Rect_functions.rect_grouper(listrec)
    #print listrec

    #for r2 in listrec:
    #    cv2.rectangle(fondo,(r2[0],r2[1]),(r2[0]+r2[2],r2[1]+r2[3]),(255,0,0),2,8,0)
    #    cv2.imwrite("imgo.jpg",fondo)


    fondo = cv2.cvtColor(fondo,cv2.CV_8U)
    (fH, fW) = fondo.shape[:2]

    #cargo el template
    template = cv2.imread("template.jpg")
    template = cv2.cvtColor(template,cv2.CV_8U)
    (tHO, tWO) = template.shape[:2]

    found = None
    for r in listrec:
        sca=0
        rx=r[0]
        ry=r[1]
        background = fondo[r[1]:(r[1]+r[3]),r[0]:(r[0]+r[2])]
        (bH, bW) = background.shape[:2]
        cv2.imwrite("imgo.jpg",background)
        if float(bH)/float(tHO)>float(bW)/float(tWO):
           sca = float(bW)/float(tWO)
        else:
          sca = float(bH)/float(tHO)

        #print scale
        if sca<=0.1:
            #print "continue"
            continue

        templatedim = cv2.resize(template, (0,0), fx=sca, fy=sca)

        for scale in np.linspace(0.1, 1.0,10)[::-1]:#resize from 0.1 to 1 template
            if scale*sca<0.1: #si la escala del template a buscar resizeado es menor al 10 % lo descarto
                continue
            resized = cv2.resize(templatedim, (0,0), fx=scale, fy=scale)
            (tH, tW) = resized.shape[:2]
            r=resized.shape[1]/templatedim.shape[1]
            if resized.shape[0] >= fH or resized.shape[1] >=fW:#si template escalado mayor q fondo
                print("Error en dimensiones")
                break
            result = cv2.matchTemplate(background,resized,cv2.TM_CCOEFF_NORMED)#busco el minimo xq es sqdiff
            (minVal, maxVal,minLoc, maxLoc) = cv2.minMaxLoc(result)
            if False:#poner en True en caso de querer ver paso a paso la deteccion del objeto y valores de matching
                # draw a bounding box around the detected region
                clone =background.copy()
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                print maxVal
                cv2.waitKey(0)
            if found is None or maxVal > found[0]:
                found = (maxVal, (maxLoc[0]+rx,maxLoc[1]+ry), tH,tW)


    (maxVal, maxLoc,tH,tW) = found
    if maxVal>0.4: #umbral para certeza de deteccion de objeto
        siguiendo = True
        # setup initial location of window
        w=int(tW)
        h=int(tH)
        r=int(maxLoc[1])#y
        c=int(maxLoc[0])#x
        track_window = (c,r,w,h)
        clone =fondo.copy()
        cv2.rectangle(clone, (c,r),(c+w,r +h), (0, 0, 255), 2)
        cv2.imshow("Objeto Encontrado", clone)
    else:
        print "Objeto no encontrado"



cv2.namedWindow("TiempoReal")
cv2.setMouseCallback("TiempoReal", click_on_mouse)

cap = cv2.VideoCapture(0)

# take first frame of the video
ret,frame = cap.read()


while(1):
    time = datetime.now()
    ret ,frame = cap.read()

    if ret == True:
        seleccion=frame.copy()
        if siguiendo==True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0,80.,80.)), np.array((255,255.,255.)))
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,255],1)
            dst=mask*dst
            kernel = np.ones((2,2),np.uint8)

            dst = cv2.erode(dst,kernel,iterations = 3)
            dst = cv2.dilate(dst,kernel,iterations = 9)
            cv2.imshow('TOTAL',255*dst)


            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            #cv2.imshow("BackProjection", dst)
            if (track_window[2]>3*track_window[3])or(track_window[3]>3*track_window[2]):#Limites de desision para dejar de seguir
                siguiendo=False

            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame,[pts],True, 255,2)
            cv2.imshow('TiempoReal',img2)
        else:
            cv2.imshow('TiempoReal',frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.imwrite("prueba.jpg",255*dst)
            break
        elif k==111:
            buscar_objeto()
        elif k==112:
            siguiendo=False

    else:
        break
    time =datetime.now()-time
    #print round(1/time.total_seconds())

cv2.destroyAllWindows()
cap.release()