# run recon.py program, control.py and the other python scripts
import subprocess
import time
import atexit

# hay que ver donde mete el stdout y stdin de los programas que abre Popen
Recon = subprocess.Popen(['python', 'Reconocimiento.py'])  # open recon script
Control = subprocess.Popen(['python','Control.py'])  # open control script

pid_Recon = Recon.pid
pid_Control = Control.pid

def cleanup():

    global Recon, Control

    Recon.terminate()
    Control.terminate()

atexit.register(cleanup)

# set each process to a specific core mask using taskset (LINUX ONLY)
#subprocess.call(["taskset", "-cp 0-3 %s" % pid_Recon])  # asigno 3 nucleos a recon
#subprocess.call(["taskset", "-cp 4 %s" % pid_Control])  # y uno solo a Control

while 1:
    time.sleep(1)

# import zmq
# import time
#
# connected = True
# port = "6001"  # connection port
#
# def control_vector(framedim,center,radius):
#     global socket, connected
#     vector = (center[0]-framedim[1]/2, framedim[0]/2-center[1], radius, time.time()*1000)
#
#     if connected: # send msg here
#         socket.send_pyobj(vector)
#
#     else:
#         print "Unable to send control vector:\n"
#         print vector
#
#     # send to control process (x, y, radio, timestamp)
#
# try:
#     context = zmq.Context()
#     socket = context.socket(zmq.PAIR)  # create a bidirectional socket
#     socket.bind("tcp://*:%s" % port)  # bind the socket to localhost:port
# except zmq.error.ZMQError:
#     print "Unable to establish socket"
#     exit()
#
# while True:
#
#     control_vector((16, 9),(1, 1),2)
#     msg = socket.recv()
#     print msg
#     time.sleep(1)