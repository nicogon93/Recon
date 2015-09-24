import zmq
import time

port = "6001"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:%s" % port)

while True:
    # non block read statement if there is nothing to read, throws a zmq.error exception
    try:
        vector = socket.recv_pyobj(zmq.NOBLOCK)

    # do something with the exception
    except zmq.error.Again as e:
        vector = e

    print vector
