# run recon program, drone control and the other APIs

# import subprocess
#
# Recon = subprocess.Popen(['python', 'Reconocimiento.py'])  # open recon script
# Control = subprocess.Popen(['python','Control.py'])  # open control script

import zmq
import time

connected = True
port = "6001"  # connection port

def control_vector(framedim,center,radius):
    global socket, connected
    vector = (center[0]-framedim[1]/2, framedim[0]/2-center[1], radius, time.time()*1000)

    if connected: # send msg here
        socket.send_pyobj(vector)

    else:
        print "Unable to send control vector:\n"
        print vector

    # send to control process (x, y, radio, timestamp)

try:
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)  # create a bidirectional socket
    socket.bind("tcp://*:%s" % port)  # bind the socket to localhost:port
except zmq.error.ZMQError:
    print "Unable to establish socket"
    exit()

while True:

    control_vector((16, 9),(1, 1),2)
    msg = socket.recv()
    print msg
    time.sleep(1)