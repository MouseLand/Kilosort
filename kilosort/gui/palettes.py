import numpy as np
from qtpy import QtGui


class DarkPalette(QtGui.QPalette):
    """Class that inherits from pyqtgraph.QtGui.QPalette and renders dark colours for the application."""

    def __init__(self):
        QtGui.QPalette.__init__(self)
        self.setup()

    def setup(self):
        self.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 27, 24))
        self.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        self.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        self.setColor(
            QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(128, 128, 128)
        )
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.ButtonText,
            QtGui.QColor(128, 128, 128),
        )
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.WindowText,
            QtGui.QColor(128, 128, 128),
        )


COLORMAP_COLORS = np.array(
    [
        [103, 0, 31],
        [112, 0, 31],
        [121, 1, 32],
        [130, 3, 33],
        [138, 5, 34],
        [146, 8, 35],
        [154, 10, 36],
        [161, 14, 38],
        [167, 17, 40],
        [173, 20, 41],
        [178, 24, 43],
        [183, 28, 45],
        [187, 34, 47],
        [191, 40, 50],
        [195, 48, 54],
        [198, 56, 57],
        [201, 64, 61],
        [204, 72, 65],
        [208, 81, 69],
        [211, 89, 73],
        [214, 96, 77],
        [217, 103, 81],
        [221, 110, 86],
        [224, 117, 91],
        [228, 125, 96],
        [231, 132, 101],
        [235, 139, 107],
        [238, 146, 112],
        [240, 152, 118],
        [242, 159, 124],
        [244, 165, 130],
        [245, 171, 136],
        [247, 177, 143],
        [248, 183, 150],
        [249, 189, 157],
        [250, 194, 164],
        [251, 200, 172],
        [251, 205, 179],
        [252, 210, 186],
        [253, 215, 193],
        [253, 219, 199],
        [253, 224, 206],
        [254, 228, 213],
        [254, 233, 220],
        [254, 238, 228],
        [254, 242, 235],
        [255, 246, 241],
        [255, 250, 247],
        [255, 253, 251],
        [255, 254, 254],
        [255, 255, 255],
        [254, 254, 254],
        [252, 252, 252],
        [249, 249, 249],
        [246, 246, 246],
        [241, 241, 241],
        [237, 237, 237],
        [232, 232, 232],
        [228, 228, 228],
        [224, 224, 224],
        [220, 220, 220],
        [217, 217, 217],
        [213, 213, 213],
        [210, 210, 210],
        [206, 206, 206],
        [202, 202, 202],
        [198, 198, 198],
        [194, 194, 194],
        [190, 190, 190],
        [186, 186, 186],
        [182, 182, 182],
        [177, 177, 177],
        [172, 172, 172],
        [167, 167, 167],
        [162, 162, 162],
        [157, 157, 157],
        [151, 151, 151],
        [146, 146, 146],
        [140, 140, 140],
        [135, 135, 135],
        [129, 129, 129],
        [124, 124, 124],
        [118, 118, 118],
        [112, 112, 112],
        [106, 106, 106],
        [100, 100, 100],
        [94, 94, 94],
        [88, 88, 88],
        [83, 83, 83],
        [77, 77, 77],
        [72, 72, 72],
        [66, 66, 66],
        [61, 61, 61],
        [56, 56, 56],
        [51, 51, 51],
        [46, 46, 46],
        [41, 41, 41],
        [36, 36, 36],
        [31, 31, 31],
        [26, 26, 26],
    ]
)

# This is matplotlib tab10 with gray moved to the last position and all 0.5 alpha
PROBE_PLOT_COLORS = (np.array(
    [
        [0.12156863, 0.46666667, 0.70588235, 0.5],
        [1.        , 0.49803922, 0.05490196, 0.5],
        [0.17254902, 0.62745098, 0.17254902, 0.5],
        [0.83921569, 0.15294118, 0.15686275, 0.5],
        [0.58039216, 0.40392157, 0.74117647, 0.5],
        [0.54901961, 0.3372549 , 0.29411765, 0.5],
        [0.89019608, 0.46666667, 0.76078431, 0.5],
        [0.7372549 , 0.74117647, 0.13333333, 0.5],
        [0.09019608, 0.74509804, 0.81176471, 0.5],
        [0.49803922, 0.49803922, 0.49803922, 0.25]
    ]
)*255).astype('int')
# Matplotlib uses float[0,1], pyqtgraph uses int[0,255]
