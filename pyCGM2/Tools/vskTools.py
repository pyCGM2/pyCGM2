# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:40:22 2017

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

from bs4 import BeautifulSoup


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
                