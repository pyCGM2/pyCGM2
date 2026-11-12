# coding: utf-8
"""Word (docx) report generation from plot images."""
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
    """Translate a label for the report (``"Clear"`` into French ``"Clair"``).

    Args:
        text (str): Label.

    Returns:
        str: Translated label.
    """

    if text == "Clear": return "Clair"
    else:
        return text

def xlsColLetterToNum(letter):
    """Convert an Excel column letter into its number (A = 1).

    Args:
        letter (str): Column letter.

    Returns:
        int: Column number.
    """
    return ord(letter)-65+1

class Report(object):
    """Word report made of analysis, intra-comparison and inter-comparison sections.

    Args:
        name (str, optional): Report name. Defaults to ``"gaitReport"``.
    """
    def __init__(self, name = "gaitReport"):
        self.name = name
        self.Analyses = []
        self.intraComparisons = []
        self.interComparisons = []


    def add_analysisSection(self, sectionTitle):
        """Add an analysis section.

        Args:
            sectionTitle (str): Title of the section.
        """

        analysis = dict()
        analysis["section"] = sectionTitle
        analysis["pages"] = list()
        analysis["pageTitles"] = list()

        self.Analyses.append(analysis)

    def add_pageToAnalysis(self, analysisIndex,page,pageTiltle):
        """Add a page (or a list of pages) to an analysis section.

        Args:
            analysisIndex (int): Index of the section (starting at 1).
            page (str | list): Image filename(s) of the page(s).
            pageTiltle (str): Page title.
        """

        self.Analyses[analysisIndex-1]["pageTitles"].append(pageTiltle)

        if isinstance(page,list):
            for pageIt in page:
                self.Analyses[analysisIndex-1]["pages"].append(pageIt)
        else:
            self.Analyses[analysisIndex-1]["pages"].append(page)

    def add_intraComparison(self, sectionTitle):
        """Add an intra-comparison section.

        Args:
            sectionTitle (str): Title of the section.
        """

        comparison = dict()
        comparison["section"] = sectionTitle
        comparison["pages"] = list()
        comparison["pageTitles"] = list()

        self.intraComparisons.append(comparison)

    def add_pageToIntraComparison(self, comparisonIndex,page,pageTiltle):
        """Add a page (or a list of pages) to an intra-comparison section.

        Args:
            analysisIndex (int): Index of the section (starting at 1).
            page (str | list): Image filename(s) of the page(s).
            pageTiltle (str): Page title.
        """

        self.intraComparisons[comparisonIndex-1]["pageTitles"].append(pageTiltle)

        if isinstance(page,list):
            for pageIt in page:
                self.intraComparisons[comparisonIndex-1]["pages"].append(pageIt)

        else:
            self.intraComparisons[comparisonIndex-1]["pages"].append(page)


    def add_interComparison(self, sectionTitle):
        """Add an inter-comparison section.

        Args:
            sectionTitle (str): Title of the section.
        """

        comparison = dict()
        comparison["section"] = sectionTitle
        comparison["pages"] = list()
        comparison["pageTitles"] = list()

        self.interComparisons.append(comparison)

    def add_pageToInterComparison(self, comparisonIndex,page,pageTiltle):
        """Add a page (or a list of pages) to an inter-comparison section.

        Args:
            analysisIndex (int): Index of the section (starting at 1).
            page (str | list): Image filename(s) of the page(s).
            pageTiltle (str): Page title.
        """

        self.interComparisons[comparisonIndex-1]["pageTitles"].append(pageTiltle)

        if isinstance(page,list):
            for pageIt in page:
                self.interComparisons[comparisonIndex-1]["pages"].append(pageIt)

        else:
            self.interComparisons[comparisonIndex-1]["pages"].append(page)


    def generateWord(self,templatePathFile, doc_path, component_path,processedInfos=None):
        """Generate the Word document from a docx template.

        Args:
            templatePathFile (str): Path of the docx template.
            doc_path (str): Path of the output document.
            component_path (str): Folder of the page images.
            processedInfos (dict, optional): Additional information passed to the template.
        """


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


