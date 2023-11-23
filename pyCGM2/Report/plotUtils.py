""" plot utils"""

def colorContext(context:str):
    """return color from event context name

    Args:
        context (str): event context

    """
    if context == "Left":
        colorContext = "red"
    elif context == "Right":
        colorContext = "blue"
    else:
        colorContext = "black"
    return colorContext
