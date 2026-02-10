import ipdb
import numpy as np
import matplotlib.pyplot as plt
from pyCGM2.Mek.lib import mekLib
import pyCGM2
LOGGER = pyCGM2.LOGGER

from pyCGM2.Cycles import cycleUtils

from pyCGM2.EMG import normalActivation

class AbstractMekPlotProcedure:
    def __init__(self):
        pass


class mekPlotTemporalEmgLayoutProcedure(AbstractMekPlotProcedure):

    def __init__(self,layout=None):
        super(mekPlotTemporalEmgLayoutProcedure, self).__init__()
        self.m_layout = layout
        self.m_legends = None

    def setData(self,groups):
        self.m_groups = groups

    def plot(self):

        if self.m_layout is not None:
    
            rows = self.m_layout['rows']
            cols = self.m_layout['cols']

            self.m_fig, self.m_axes = plt.subplots(rows, cols, figsize=(8.27,11.69), dpi=100, facecolor="white")
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


            self.m_fig.suptitle(self.m_layout['sup_title'], fontsize=14)

            for ax in self.m_axes.flat:
                ax.set_ylabel("angle (deg)",size=8)
                ax.tick_params(axis='x', which='major', labelsize=6)
                ax.tick_params(axis='y', which='major', labelsize=6)

                        
            for i, plot_cfg in enumerate(self.m_layout['plots']):
                row, col = plot_cfg['position']
                if rows == 1 and cols == 1:
                    ax = self.m_axes
                elif rows == 1 or cols == 1:
                    ax = self.m_axes[max(row, col)]
                else:
                    ax = self.m_axes[row][col]

                
                for curve in plot_cfg.get('curves', []):
                    eventContext =  curve.get('eventContext') 
                    label_data = curve['data']
                    if "." in label_data:
                        label, axis_str = label_data.split(".")
                    else:
                        label = label_data
                        axis_str = "0"

    
                    axis = int(axis_str)
                    legend_label0 = curve.get('legendLabel')

                    data = self.m_groups.retrieve_set(label).read()[:,axis]

                    start_time = self.m_groups.retrieve_set(label).retrieve_attribute("StartTime").read()
                    sample_rate = self.m_groups.retrieve_set(label).retrieve_attribute("SampleRate").read()

                    ndata = data.shape[0]

                    x = start_time + np.arange(ndata) / sample_rate

                    if eventContext=="Left": color="red"
                    if eventContext=="Right": color="blue"

                    ax.plot(x,data, label="", color=color)
                    lfs = self.m_groups.retrieve_group("events/Foot Strike").retrieve_set("Left").read()
                    lfo = self.m_groups.retrieve_group("events/Foot Off").retrieve_set("Left").read()

                    for timeIt in  lfs:
                        ax.axvline( x= timeIt, color = "red", linestyle = "-")
                    for timeIt in  lfo:
                        ax.axvline( x= timeIt, color = "red", linestyle = "--")
                    
                    rfs = self.m_groups.retrieve_group("events/Foot Strike").retrieve_set("Right").read()
                    rfo = self.m_groups.retrieve_group("events/Foot Off").retrieve_set("Right").read()

                    for timeIt in  rfs:
                        ax.axvline( x= timeIt, color = "blue", linestyle = "-")

                    for timeIt in  rfo:
                        ax.axvline( x= timeIt, color = "blue", linestyle = "--")

                    if eventContext=="Left":
                        leftCycles = cycleUtils.build_cycles_classified(lfs, lfo, None, None)
                        for cycleIt  in leftCycles:
                            pos,burstDuration=normalActivation.getNormalBurstActivityTime(plot_cfg['normalActivation'],cycleIt["start"], cycleIt["footOff"], cycleIt["end"])
                            for j in range(0,len(pos)):
                                ax.add_patch(plt.Rectangle((pos[j],0),burstDuration[j],ax.get_ylim()[1] , color='g',alpha=0.1))

                    if eventContext=="Right":
                        rightCycles = cycleUtils.build_cycles_classified(rfs, rfo, None, None)

                        for cycleIt  in rightCycles:
                            pos,burstDuration=normalActivation.getNormalBurstActivityTime(plot_cfg['normalActivation'],cycleIt["start"], cycleIt["footOff"], cycleIt["end"])
                            for j in range(0,len(pos)):
                                ax.add_patch(plt.Rectangle((pos[j],0),burstDuration[j],ax.get_ylim()[1] , color='g',alpha=0.1))


                ax.set_title(plot_cfg['title'],size=8)
                ax.set_ylabel(plot_cfg['ylabel'],size=8)

                if plot_cfg['ylim'][0]=="-inf":plot_cfg['ylim'][0] = min(data)
                if plot_cfg['ylim'][1]=="inf":plot_cfg['ylim'][1] = max(data)

                ax.set_ylim(plot_cfg['ylim'])
                
                if i==0:  ax.legend(fontsize=6, loc="upper right")

            plt.tight_layout()
            # plt.show()




class mekPlotSingleGroupLayoutProcedure(AbstractMekPlotProcedure):

    def __init__(self,layout=None, consistency=False):
        super(mekPlotSingleGroupLayoutProcedure, self).__init__()
        self.m_layout = layout
        self.m_consistency = consistency

        self.m_legends = None
        self.m_groups = None

    def filterEventContext(self, eventContext=None):
        def remove_curves(config, eventContextToRemove):
            for plot in config.get("plots", []):
                curves = plot.get("curves", [])
                plot["curves"] = [c for c in curves if c.get("eventContext") !=eventContextToRemove]
            return config

        if eventContext == "Left": 
            self.m_layout = remove_curves(self.m_layout, "Right")
        if eventContext == "Right": 
            self.m_layout = remove_curves(self.m_layout, "Left")



    def setData(self,group):
            self.m_group = group


    def plot(self):

        if self.m_layout is not None:
    
            rows = self.m_layout['rows']
            cols = self.m_layout['cols']

            self.m_fig, self.m_axes = plt.subplots(rows, cols, figsize=(8.27,11.69), dpi=100, facecolor="white")
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

            self.m_fig.suptitle(self.m_layout['sup_title'], fontsize=14)

            for ax in self.m_axes.flat:
                ax.set_ylabel("angle (deg)",size=8)
                ax.tick_params(axis='x', which='major', labelsize=6)
                ax.tick_params(axis='y', which='major', labelsize=6)

                # ax.set_xlabel("Cycle %",size=8)

            
            for i, plot_cfg in enumerate(self.m_layout['plots']):
                row, col = plot_cfg['position']
                if rows == 1 and cols == 1:
                    ax = self.m_axes
                elif rows == 1 or cols == 1:
                    ax = self.m_axes[max(row, col)]
                else:
                    ax = self.m_axes[row][col]

                legend_used = []
                for curve in plot_cfg.get('curves', []):
                    label_data = curve['data']
                    if "." in label_data:
                        label, axis_str = label_data.split(".")
                    else:
                        label = label_data
                        axis_str = "0"

    
                    axis = int(axis_str)
                    legend_label = curve.get('legendLabel')
                    

                    if curve.get('eventContext') == "Left": color = 'red'
                    if curve.get('eventContext') == "Right": color = 'blue'

                    try:

                        cycleValues ,attrs = mekLib.gatherCycle(self.m_group, label)

                        footOffs = np.array([float(attr["footOff"]) for attr in attrs if "footOff" in attr])
                        controlateral_footOff = np.array([float(attr["controlateral_footOff"]) for attr in attrs if "controlateral_footOff" in attr])
                        controlateral_footStrike = np.array([float(attr["controlateral_footStrike"]) for attr in attrs if "controlateral_footStrike" in attr])


                        if self.m_consistency:
                            for cycleIndex in range(cycleValues.shape[0]):
                                if cycleIndex>0: legend_label="_nolegend_"
                                data = cycleValues[cycleIndex, :, axis]
                                ax.plot(data, label=legend_label, color=color)
                                ax.axvline( x= footOffs[cycleIndex], color=color , linestyle = "--")
                                ax.axvline(controlateral_footOff[cycleIndex],ymin=0.9, ymax=1.0,color=color,ls='dotted')
                                ax.axvline(controlateral_footStrike[cycleIndex],ymin=0.9, ymax=1.0,color=color,ls='dotted')

                        else:
                            data = cycleValues[:, :, axis].mean(axis=0)
                            std = cycleValues[:, :, axis].std(axis=0)

                            if curve.get('eventContext') == "Left": color = 'red'
                            if curve.get('eventContext') == "Right": color = 'blue'


                            ax.plot(data, label=legend_label, color=color)
                            ax.fill_between(np.linspace(0,data.shape[0]-1,data.shape[0]), data-std, data+std, facecolor=color, alpha=0.5,linewidth=0)

                            ax.axvline( x= footOffs.mean(), color=color , linestyle = "--")
                            ax.axvline(controlateral_footOff.mean(),ymin=0.9, ymax=1.0,color=color,ls='dotted')
                            ax.axvline(controlateral_footStrike.mean(),ymin=0.9, ymax=1.0,color=color,ls='dotted')


                    except ValueError:
                        LOGGER.logger.warning(f"Label {label} not found in the data group.")
                        pass
                         

                ax.set_title(plot_cfg['title'],size=8)
                ax.set_ylabel(plot_cfg['ylabel'],size=8)
                ax.set_ylim(plot_cfg['ylim'])
                
                if i==0:  ax.legend(fontsize=6, loc="upper right")

            plt.tight_layout()
            # plt.show()
        


# class mekPlotLayoutProcedure(AbstractMekPlotProcedure):

#     def __init__(self,layout=None, consistency=False):
#         super(mekPlotLayoutProcedure, self).__init__()
#         self.m_layout = layout
#         self.m_consistency = consistency

#         self.m_legends = None
#         self.m_groups = None

#     def filterEventContext(self, eventContext=None):
#         def remove_curves(config, eventContextToRemove):
#             for plot in config.get("plots", []):
#                 curves = plot.get("curves", [])
#                 plot["curves"] = [c for c in curves if c.get("eventContext") !=eventContextToRemove]
#             return config

#         if eventContext == "Left": 
#             self.m_layout = remove_curves(self.m_layout, "Right")
#         if eventContext == "Right": 
#             self.m_layout = remove_curves(self.m_layout, "Left")



#     def setData(self,groups,legends=[]):
#         if not isinstance(groups,list):
#             self.m_groups = [groups]
#         else:
#             self.m_groups = groups

#         if legends!=[]:
#             if len(legends)==len(self.m_groups):
#                 self.m_legends = legends
#             else :
#                 raise ValueError("Length of legends must match length of groups.")




#     def plot(self):

#         if self.m_layout is not None:
    
#             rows = self.m_layout['rows']
#             cols = self.m_layout['cols']

#             self.m_fig, self.m_axes = plt.subplots(rows, cols, figsize=(8.27,11.69), dpi=100, facecolor="white")
#             plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


#             if len(self.m_groups)>1:
#                 colormap_i_blue=[ plt.cm.Blues(i) for i in np.linspace(0.2, 1, len(self.m_groups))]
#                 colormap_i_blue = [(0,0,0)] + colormap_i_blue
#                 colormap_i_red=[ plt.cm.Reds(i) for i in np.linspace(0.2, 1, len(self.m_groups))]
#                 colormap_i_red = [(0,0,0)] + colormap_i_red

#             self.m_fig.suptitle(self.m_layout['sup_title'], fontsize=14)

#             for ax in self.m_axes.flat:
#                 ax.set_ylabel("angle (deg)",size=8)
#                 ax.tick_params(axis='x', which='major', labelsize=6)
#                 ax.tick_params(axis='y', which='major', labelsize=6)

#                 # ax.set_xlabel("Cycle %",size=8)

            
#             for i, plot_cfg in enumerate(self.m_layout['plots']):
#                 row, col = plot_cfg['position']
#                 if rows == 1 and cols == 1:
#                     ax = self.m_axes
#                 elif rows == 1 or cols == 1:
#                     ax = self.m_axes[max(row, col)]
#                 else:
#                     ax = self.m_axes[row][col]

#                 legend_used = []
#                 for curve in plot_cfg.get('curves', []):
#                     label_data = curve['data']
#                     if "." in label_data:
#                         label, axis_str = label_data.split(".")
#                     else:
#                         label = label_data
#                         axis_str = "0"

    
#                     axis = int(axis_str)
#                     legend_label0 = curve.get('legendLabel')

 
#                     try:
#                         valuesByGroup = []
#                         footOffByGroup = []
#                         for group in self.m_groups:
#                             array3D ,attrs = mekLib.gatherCycle(group, label)
#                             valuesByGroup.append(array3D)

#                             footOff_array = np.array([float(attr["footOff"]) for attr in attrs if "footOff" in attr])
#                             footOffByGroup.append(footOff_array)

#                         if self.m_consistency:
#                             for j in range(len(valuesByGroup)):


#                                 if len(valuesByGroup)==1:                            
#                                     if curve.get('eventContext') == "Left": color = 'red'
#                                     if curve.get('eventContext') == "Right": color = 'blue'
#                                 else:
#                                     if curve.get('eventContext') == "Left": color = colormap_i_red[j]
#                                     if curve.get('eventContext') == "Right": color = colormap_i_blue[j]

#                                 for cycleIndex in range(valuesByGroup[j].shape[0]):
#                                     data = valuesByGroup[j][cycleIndex, :, axis]


#                                     if self.m_legends is not None :
#                                         legend_label = f"{legend_label0} - {self.m_legends[j]}"
#                                     else:
#                                         legend_label = legend_label0


#                                     if legend_label not in legend_used:
#                                         ax.plot(data, label=legend_label, color=color)
#                                         legend_used.append(legend_label)
#                                     else:
#                                         ax.plot(data, label="_nolegend_", color=color)

#                         else:
#                             for j, arr in enumerate(valuesByGroup):
#                                 data = valuesByGroup[j][:, :, axis].mean(axis=0)
#                                 std = valuesByGroup[j][:, :, axis].std(axis=0)

#                                 import ipdb; ipdb.set_trace()

#                                 if len(valuesByGroup)==1:                            
#                                     if curve.get('eventContext') == "Left": color = 'red'
#                                     if curve.get('eventContext') == "Right": color = 'blue'
#                                 else:
#                                     if curve.get('eventContext') == "Left": color = colormap_i_red[j]
#                                     if curve.get('eventContext') == "Right": color = colormap_i_blue[j]

#                                 if self.m_legends is not None :
#                                         legend_label = f"{legend_label0} - {self.m_legends[j]}"
#                                 else:
#                                     legend_label = legend_label0

#                                 # import ipdb; ipdb.set_trace()
#                                 if legend_label not in legend_used:
#                                         ax.plot(data, label=legend_label, color=color)
#                                         legend_used.append(legend_label)
#                                 else:
#                                     ax.plot(data, label="_nolegend_", color=color)

#                                 ax.fill_between(np.linspace(0,data.shape[0]-1,data.shape[0]), data-std, data+std, facecolor=color, alpha=0.5,linewidth=0)

#                     except ValueError:
#                         LOGGER.logger.warning(f"Label {label} not found in the data group.")
#                         pass
                         

#                 ax.set_title(plot_cfg['title'],size=8)
#                 ax.set_ylabel(plot_cfg['ylabel'],size=8)
#                 ax.set_ylim(plot_cfg['ylim'])
                
#                 if i==0:  ax.legend(fontsize=6, loc="upper right")

#             plt.tight_layout()
#             # plt.show()
        
        



class mekPlotFilter:
    def __init__(self,procedure=None):

        self.m_procedure = procedure

    def run(self):
        if self.m_procedure is not None:
            self.m_procedure.plot()
            plt.show()
