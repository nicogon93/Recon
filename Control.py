import zmq
import time

debug=False

port = "6002"
vector=0
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:%s" % port)

available = True
data_in = []

while True:

    while available:
        # non blocking read statement if there is nothing to read, throws a zmq.error exception
        # read entire buffer, keep the last value
        try:
            vector = socket.recv_pyobj(zmq.NOBLOCK)
            # if this line is executed it means that there is something of value in vector
            data_in = vector

        # do something with the exception
        except zmq.error.Again as e:
            # nothing else to read
            available = False

    available = True  # reset loop for next iteration

    if debug:
        print "----------------Vector recibido----------------\n"
        print vector
    time.sleep(2)

# cleanup before exit
socket.disconnect()
