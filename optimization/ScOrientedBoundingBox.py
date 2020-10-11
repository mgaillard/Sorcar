import bpy

class ScOrientedBoundingBox:

    def __init__(self, obj):
        self.center = obj.location.copy()
        self.dimensions = obj.dimensions.copy()


    def is_equal(self, other):
        return self.center == other.center and self.dimensions == other.dimensions


    def list_points_to_match(self):
        min_corner = self.center - self.dimensions / 2.0
        max_corner = self.center + self.dimensions / 2.0
        return [min_corner, max_corner]


    def __repr__(self):
        return "OBB(Center:{}, Dimensions:{})".format(self.center, self.dimensions)
