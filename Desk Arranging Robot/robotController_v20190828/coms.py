import serial
import logging
import threading
import concurrent.futures

# if received execution done signal 'E', write messageline is_done True
# if new command string put to messageline, write messageline is_ready True
# if command line is sent, make is_done and is_ready False

class messageLine():
    def __init__(self):
        self.message = ""
        self.is_done = False
        self.is_ready = False
        self.event = threading.Event()
        self._lock = threading.Lock()

def sendMessage(msgLine, ser):
    while True:
        if msgLine.is_done and msgLine.is_ready:
            with msgLine._lock:
                ser.write(msgLine.message.encode())
                logging.info("Message sent: %s", msgLine.message[0:-1])

                msgLine.is_done = False
                msgLine.is_ready = False

        if msgLine.event.is_set():
            break

def receiveMessage(msgLine, ser):
    while True:
        if ser.in_waiting:
            with msgLine._lock:
                message = ser.readline().decode()[0:-2]
                logging.info("Message received: %s", message)
                
                if message == "Initialized" or len(message) == 1 and message[0] == 'E':
                    msgLine.is_done = True
                elif len(message) == 1 and message[0] == 'T':
                    break

    logging.info("Terminating serial communication")
    msgLine.event.set()

def putMessage(msgLine, string):
    msgLine.message = string + '\n'
    msgLine.is_ready = True

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format = format, level = logging.INFO, datefmt="%H:%M:%S")

    arduino = serial.Serial('COM7', 9600)
    logging.info("Initializing arduino")

    msg_line = messageLine()

    with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
        executor.submit(receiveMessage, msg_line, arduino)
        executor.submit(sendMessage, msg_line, arduino)

        while True:
            string = input()
            putMessage(msg_line, string)

