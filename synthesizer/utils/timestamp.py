from datetime import datetime


def timestamp():
    return datetime.now().strftime("%m-%d %H:%M:%S")
