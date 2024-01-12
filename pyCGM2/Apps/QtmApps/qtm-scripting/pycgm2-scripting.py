import qtm

import os
import json


menu_id = qtm.gui.insert_menu_submenu(None, "pyCGM2 menu")

def setDataPath(path):
    os.chdir(path)

def setTrialName(trialname):
    globals()["TrialName"] = trialname

def getDataPath():
    print(os.getcwd())

def getTrialName():
    print(globals()["TrialName"])


def _updateEvent():
    # event = {"label": "New Event", "time": 0.0, "color": 255}
    # qtm.data.object.event.add_event(event)
    
    path = os.getcwd()
    print(os.getcwd())
    print("Hello world!")

    fichiers_json = [f for f in os.listdir(path) if f.endswith('.json')]
    
    filename = globals()["TrialName"] + "-events.json"
    events = json.loads(open((path+"/"+filename)).read())
    print(events)
    
    for it in events["Events"]["LeftFootStrike"]:
        color = qtm.utilities.color.rgb(1, 0, 0)
        event = {"label": "Left Foot Strike", "time": it/100, "color": color}
        qtm.data.object.event.add_event(event)

    for it in events["Events"]["LeftFootOff"]:
        color = qtm.utilities.color.rgb(1, 0.6, 0.6)
        event = {"label": "Left Foot Off", "time": it/100, "color": color}
        qtm.data.object.event.add_event(event)

    for it in events["Events"]["RightFootStrike"]:
        color = qtm.utilities.color.rgb(0, 0, 1)
        event = {"label": "Right Foot Strike", "time": it/100, "color": color}
        qtm.data.object.event.add_event(event)

    for it in events["Events"]["RightFootOff"]:
        color = qtm.utilities.color.rgb(0.6, 0.6, 1)
        event = {"label": "Right Foot Off", "time": it/100, "color": color}
        qtm.data.object.event.add_event(event)


   


qtm.gui.add_command("events")
qtm.gui.set_command_execute_function("events", _updateEvent)
qtm.gui.insert_menu_button(menu_id, "Update Events", "events")


#printing_sub_menu_id = qtm.gui.insert_menu_submenu(menu_id, "Select File")

