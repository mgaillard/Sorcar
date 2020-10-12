import bpy

import mathutils

class ScOrientedBoundingBox:

    def __init__(self, center, axis, extent):
        self.center = center
        self.axis = axis
        self.extent = extent

    @classmethod
    def fromObject(cls, obj):
        matrix = obj.matrix_world
        center = mathutils.Vector((matrix[0][3], matrix[1][3], matrix[2][3]))
        axis = [
            mathutils.Vector((matrix[0][0], matrix[1][0], matrix[2][0])).normalized(),
            mathutils.Vector((matrix[0][1], matrix[1][1], matrix[2][1])).normalized(),
            mathutils.Vector((matrix[0][2], matrix[1][2], matrix[2][2])).normalized()
        ]
        extent = obj.dimensions / 2.0

        return cls(center, axis, extent)


    @classmethod
    def defaultBoundingBox(cls):
        center = mathutils.Vector((0.0, 0.0, 0.0))
        axis = [
            mathutils.Vector((1.0, 0.0, 0.0)),
            mathutils.Vector((0.0, 1.0, 0.0)),
            mathutils.Vector((0.0, 0.0, 1.0))
        ]
        extent = mathutils.Vector((0.0, 0.0, 0.0))

        return cls(center, axis, extent)


    def is_equal(self, other):
        return (self.center == other.center and
                self.axis[0] == other.axis[0] and
                self.axis[1] == other.axis[1] and
                self.axis[2] == other.axis[2] and
                self.extent == other.extent)


    def list_points_to_match(self):
        return [
            self.center - self.extent[0] * self.axis[0] - self.extent[1] * self.axis[1] - self.extent[2] * self.axis[2],
            self.center + self.extent[0] * self.axis[0] - self.extent[1] * self.axis[1] - self.extent[2] * self.axis[2],
            self.center - self.extent[0] * self.axis[0] + self.extent[1] * self.axis[1] - self.extent[2] * self.axis[2],
            self.center - self.extent[0] * self.axis[0] - self.extent[1] * self.axis[1] + self.extent[2] * self.axis[2],
            self.center + self.extent[0] * self.axis[0] + self.extent[1] * self.axis[1] - self.extent[2] * self.axis[2],
            self.center + self.extent[0] * self.axis[0] - self.extent[1] * self.axis[1] + self.extent[2] * self.axis[2],
            self.center - self.extent[0] * self.axis[0] + self.extent[1] * self.axis[1] + self.extent[2] * self.axis[2],
            self.center + self.extent[0] * self.axis[0] + self.extent[1] * self.axis[1] + self.extent[2] * self.axis[2]
        ]


    def __repr__(self):
        return "OBB(Center:{}, Axis: {}, {}, {}, Extent:{})".format(self.center, self.axis[0], self.axis[1], self.axis[2], self.extent)
