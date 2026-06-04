import colorsys
from typing import List


def set_turning_leds(direction: str) -> dict:
    """Set LEDs to indicate turning direction."""
    off = [0.0] * 3
    white = [1.0] * 3
    yellow = [1.0, 1.0, 0.0]
    red = [1.0, 0.0, 0.0]
    d = {0:off, 2:off, 3:off, 4:off}

    match direction: 
        case "right": 
            d[2] = yellow
            d[4] = yellow
        case "left":
            d[0] = yellow
            d[3] = yellow
        case "forward":
            d[0] = white
            d[2] = white
        case "down": 
            d[3] = red
            d[4] = red
        case _:
            pass
            
    return d 