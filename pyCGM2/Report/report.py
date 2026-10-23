# coding: utf-8
import copy
import numpy as np

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files
import companion

from jinja2 import Environment, PackageLoader, select_autoescape,FileSystemLoader
from dataclasses import dataclass


from docx.shared import Cm
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm, Inches, Mm, Emu
import random
import datetime
import matplotlib.pyplot as plt
import os
import docx
from docx import Document, text
from docx.shared import Pt
from docx.shared import RGBColor
import pandas as pd
from openpyxl import load_workbook
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml




def translate(text):

    if text == "Clear": return "Clair"
    else:
        return text

def xlsColLetterToNum(letter):
    return ord(letter)-65+1

class Report(object):
    def __init__(self, name = "gaitReport"):
        self.name = name
        self.Analyses = []
        self.intraComparisons = []
        self.interComparisons = []


    def add_analysisSection(self, sectionTitle):

        analysis = dict()
        analysis["section"] = sectionTitle
        analysis["pages"] = list()
        analysis["pageTitles"] = list()

        self.Analyses.append(analysis)

    def add_pageToAnalysis(self, analysisIndex,page,pageTiltle):

        self.Analyses[analysisIndex-1]["pageTitles"].append(pageTiltle)

        if isinstance(page,list):
            for pageIt in page:
                self.Analyses[analysisIndex-1]["pages"].append(pageIt)
        else:
            self.Analyses[analysisIndex-1]["pages"].append(page)

    def add_intraComparison(self, sectionTitle):

        comparison = dict()
        comparison["section"] = sectionTitle
        comparison["pages"] = list()
        comparison["pageTitles"] = list()

        self.intraComparisons.append(comparison)

    def add_pageToIntraComparison(self, comparisonIndex,page,pageTiltle):

        self.intraComparisons[comparisonIndex-1]["pageTitles"].append(pageTiltle)

        if isinstance(page,list):
            for pageIt in page:
                self.intraComparisons[comparisonIndex-1]["pages"].append(pageIt)

        else:
            self.intraComparisons[comparisonIndex-1]["pages"].append(page)


    def add_interComparison(self, sectionTitle):

        comparison = dict()
        comparison["section"] = sectionTitle
        comparison["pages"] = list()
        comparison["pageTitles"] = list()

        self.interComparisons.append(comparison)

    def add_pageToInterComparison(self, comparisonIndex,page,pageTiltle):

        self.interComparisons[comparisonIndex-1]["pageTitles"].append(pageTiltle)

        if isinstance(page,list):
            for pageIt in page:
                self.interComparisons[comparisonIndex-1]["pages"].append(pageIt)

        else:
            self.interComparisons[comparisonIndex-1]["pages"].append(page)


    def generateWord(self,templatePathFile, doc_path, component_path,processedInfos=None):


        template = DocxTemplate(templatePathFile)

        for analysisIt in self.Analyses:
            for index in range(0,len(analysisIt["pages"])):
                name = component_path + analysisIt["pages"][index]
                analysisIt["pages"][index] = InlineImage(template,name,Cm(14),Cm(19))

        for analysisIt in self.intraComparisons:
            for index in range(0,len(analysisIt["pages"])):
                name = component_path + analysisIt["pages"][index]
                analysisIt["pages"][index] = InlineImage(template,name,Cm(14),Cm(19))

        for analysisIt in self.interComparisons:
            for index in range(0,len(analysisIt["pages"])):
                name = component_path + analysisIt["pages"][index]
                analysisIt["pages"][index] = InlineImage(template,name,Cm(14),Cm(19))


        #Render automated report
        data={"Report": self, "infos":processedInfos}
        template.render(data)
        template.save(doc_path+'gaitReport.docx')


