  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
###############################################################################
#                                                                             #
# file:    4_multi_core.py                                                    #
#                                                                             #
# authors: Andre Heil  - avh34                                                #
#          Jingyao Ren - jr386                                                #
#                                                                             #
# date:    December 1st 2015                                                  #
#                                                                             #
# brief:   This file uses multicore processing to track your face. This is    #
#          similar to 1_single_core.py except now we utilize all four cores   #
#          to create a more fluid video.                                      #
#                                                                             #
###############################################################################


### Imports ###################################################################

from picamera.array import PiRGBArray
from picamera import PiCamera
from functools import partial

import multiprocessing as mp
import cv2
#import os
import time


### Setup #####################################################################

#os.putenv( 'SDL_FBDEV', '/dev/fb0' )

#resX = 320
#resY = 240

#cx = resX / 2
#cy = resY / 2

#os.system( "echo 0=150 > /dev/servoblaster" )
#os.system( "echo 1=150 > /dev/servoblaster" )

#xdeg = 150
#ydeg = 150


# Setup the camera
camera = PiCamera()
camera.resolution = ( resX, resY )
camera.framerate = 60

# Use this as our output
rawCapture = PiRGBArray( camera, size=( resX, resY ) )

# The face cascade file to be used
face_cascade = cv2.CascadeClassifier('/home/pi/opencv-2.4.9/data/lbpcascades/lbpcascade_frontalface.xml')

t_start = time.time()
fps = 0


### Helper Functions ##########################################################

def get_faces( img ):
    
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    faces = face_cascade.detectMultiScale( gray )
        
    return faces, img

def draw_frame( img, faces ):

    global xdeg
    global ydeg
    global fps
    global time_t

    # Draw a rectangle around every face
    for ( x, y, w, h ) in faces:

        cv2.rectangle( img, ( x, y ),( x + w, y + h ), ( 200, 255, 0 ), 2 )
        cv2.putText(img, "Face No." + str( len( faces ) ), ( x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )

        tx = x + w/2
        ty = y + h/2
        
     #   if   ( cx - tx > 15 and xdeg <= 190 ):
     #       xdeg += 1
     #       os.system( "echo 0=" + str( xdeg ) + " > /dev/servoblaster" )
     #   elif ( cx - tx < -15 and xdeg >= 110 ):
     #       xdeg -= 1
     #       os.system( "echo 0=" + str( xdeg ) + " > /dev/servoblaster" )

     #   if   ( cy - ty > 15 and ydeg >= 110 ):
     #       ydeg -= 1
     #       os.system( "echo 1=" + str( ydeg ) + " > /dev/servoblaster" )
     #   elif ( cy - ty < -15 and ydeg <= 190 ):
     #       ydeg += 1
     #       os.system( "echo 1=" + str( ydeg ) + " > /dev/servoblaster" )
    
    # Calculate and show the FPS
    fps = fps + 1
    sfps = fps / (time.time() - t_start)
    cv2.putText(img, "FPS : " + str( int( sfps ) ), ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 ) 
    
    cv2.imshow( "Frame", img )
    cv2.waitKey( 1 )


### Main ######################################################################

if __name__ == '__main__':

    pool = mp.Pool( processes=4 )
    fcount = 0
    
    camera.capture( rawCapture, format="bgr" )  

    r1 = pool.apply_async( get_faces, [ rawCapture.array ] )    
    r2 = pool.apply_async( get_faces, [ rawCapture.array ] )    
    r3 = pool.apply_async( get_faces, [ rawCapture.array ] )    
    r4 = pool.apply_async( get_faces, [ rawCapture.array ] )    

    f1, i1 = r1.get()
    f2, i2 = r2.get()
    f3, i3 = r3.get()
    f4, i4 = r4.get()

    rawCapture.truncate( 0 )    

    for frame in camera.capture_continuous( rawCapture, format="bgr", use_video_port=True ):
        image = frame.array

        if   fcount == 1:
            r1 = pool.apply_async( get_faces, [ image ] )
            f2, i2 = r2.get()
            draw_frame( i2, f2 )

        elif fcount == 2:
            r2 = pool.apply_async( get_faces, [ image ] )
            f3, i3 = r3.get()
            draw_frame( i3, f3 )

        elif fcount == 3:
            r3 = pool.apply_async( get_faces, [ image ] )
            f4, i4 = r4.get()
            draw_frame( i4, f4 )

        elif fcount == 4:
            r4 = pool.apply_async( get_faces, [ image ] )
            f1, i1 = r1.get()
            draw_frame( i1, f1 )

            fcount = 0

        fcount += 1
        
        rawCapture.truncate( 0 )
