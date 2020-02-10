"""
class position to create a (lat, long) object
"""


class position:
    def __init__(self, lat, lon):
        self.latitude = lat
        self.longitude = lon

    def __str__(self):
        return "(" + str(self.latitude) + ", " + str(self.longitude) + ")"
