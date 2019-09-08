import serial
import time
import logging

def sendReceive(ser, msg):
    # add message end charactor '\0' and send it to arduino
    msg = msg + '\0'
    ser.write(msg.encode())
    logging.info("MESSAGE SENT: %s", msg[:-1])

    while True:
        msg = ser.readline().decode()[:-2]
        if msg == "DONE":
            logging.info("MESSAGE RECEIVED: %s", msg)
            break
        else:
            logging.info("MESSAGE RECEIVED: %s", msg)

if __name__ ==  "__main__":

    # logging format configuration
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format = format, level = logging.INFO, datefmt="%H:%M:%S")

    # establish serial communication with arduino
    ser = serial.Serial('COM7', 9600)
    time.sleep(2)
    logging.info("SERIAL COMMUNICATION INITIALIZED")

    while True:
        message = input()
        sendReceive(ser, message)
