from __future__ import absolute_import


from models.baseline import VideoBaseline
from models.net.models import CNN

__factory = {
    'video_baseline': VideoBaseline,
    'NVAN' : CNN
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
