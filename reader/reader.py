import wget
from os.path import exists

from config import ROBERT_FROST_TXT_PATH


class Reader:

    def __init__(self, robertFrostUrl):
        if not exists(ROBERT_FROST_TXT_PATH):
            self.robertFrostFile = wget.download(robertFrostUrl)
        else:
            self.robertFrostFile = ROBERT_FROST_TXT_PATH

    def readRobertFrostTxt(self):
        robertFrostLineList = []
        with open(self.robertFrostFile, 'r') as rf:
            for line in rf:
                robertFrostLineList.append(line)
        return robertFrostLineList
