import os


def formatSize(bytes):
    try:
        kb = float(bytes)
        kb = kb / 1024
    except:
        return "Error"
    return kb


def getFileSize(path):
    try:
        size = os.path.getsize(path)
        return formatSize(size)
    except Exception as err:
        print(err)


def getDirSize(path):
    sumsize = 0
    try:
        filename = os.walk(path)
        for root, dirs, files in filename:
            for fle in files:
                size = os.path.getsize(path + fle)
                sumsize += size
        return formatSize(sumsize)
    except Exception as err:
        print(err)
