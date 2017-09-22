# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import string
import logging


def checkSetReadOnly(vskfilename):
    file0 = open(vskfilename,'r')
    content = file0.read()

    flag=True  if content.find('READONLY="true"') !=-1 else False
    print flag

    file0.close()

    if flag:
        logging.warning("read Only found")
        content2 = string.replace(content, 'READONLY="true"', 'READONLY="false"')

        with open(vskfilename, "w") as text_file:
            text_file.write(content2)



class Vsk(object):
    """

    """

    def __init__(self,file):

        self.m_file=file

        infile = open(file,"r")
        contents = infile.read()
        soup = BeautifulSoup(contents,'xml')

        self.m_soup = soup



    def getStaticParameterValue(self, label):

        staticParameters = self.m_soup.find_all('StaticParameter')
        for sp in staticParameters:
            if sp.attrs["NAME"] == label:
                return sp.attrs["VALUE"]
