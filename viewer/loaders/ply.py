import numpy as np
from plyfile import PlyData


def load_ply_file(path):
    """Load a PLY file and return a dict of vertex properties as numpy arrays."""
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    data = {}
    for prop in vertex.properties:
        name = prop.name
        data[name] = np.array(vertex[name])
    return data
