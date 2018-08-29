from dbse.tools.Singleton import Singleton
import time
import datetime


class Logging(Singleton):
    OFF = False
    
    @staticmethod
    # define Log function
    def log(text):
        if Logging.OFF: return
        print(time.strftime('%Y.%m.%d, %H:%M:%S') + ': ' + text)

    @staticmethod
    def disable():
        Logging.OFF = True
        
    @staticmethod
    def ping():
        return datetime.datetime.now()

    @staticmethod
    def pong(dt):
        now = datetime.datetime.now()
        diff = now - dt
        ms = round(diff.total_seconds() * 1000)
        return ms
