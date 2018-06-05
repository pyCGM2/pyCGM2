# -*- coding: utf-8 -*-
import pandas as pd

class Annotator(object):

    def __init__(self,fig):
        self.annotations = list()
        self.fig = fig
        self.count = 0

    def IncreasedRange(self,axIndex,frame,min,max, context, timing="Throughout Cycle"):

        if context == "Left":
            color = "red"
        elif context == "Right":
            color = "blue"
        elif context == "Bilateral":
            color = "black"
        else:
            color = "grey"

        key = self.count+1

        self.fig.axes[axIndex].annotate("", xy=(frame, min), xycoords='data',
                xytext=(frame, max), textcoords='data',va='center', arrowprops=dict(arrowstyle="<->",color=color))
        self.fig.axes[axIndex].text(frame+3, (min+max)/2.0,key , ha="center", va="center", rotation=0, size=8,color=color)

        d = {'Letter': [key], 'Type': ["Increased"], 'Side':[context], 'Variable' :[self.fig.axes[axIndex].get_title()] , "Timing":[timing]}
        df = pd.DataFrame(data=d)

        self.count+=1

        self.annotations.append(df)


    def getAnnotations(self):
        dataframe = pd.concat(self.annotations,ignore_index=True)

        return dataframe
