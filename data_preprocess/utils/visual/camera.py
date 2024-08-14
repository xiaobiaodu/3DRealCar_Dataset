import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CameraPoseVisualizer:
    def __init__(self, xlim=None, ylim=None, zlim=None):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_axes(Axes3D(self.fig))
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=0.2, aspect_ratio=0.3, name=None):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio * 2, -focal_len_scaled * aspect_ratio, focal_len_scaled*1, 1],
                               [focal_len_scaled * aspect_ratio * 2, focal_len_scaled * aspect_ratio, focal_len_scaled*1, 1],
                               [-focal_len_scaled * aspect_ratio * 2, focal_len_scaled * aspect_ratio, focal_len_scaled*1, 1],
                               [-focal_len_scaled * aspect_ratio * 2, -focal_len_scaled * aspect_ratio, focal_len_scaled*1, 1]])
        extrinsic_righthand = extrinsic.copy()
        vertex_transformed = vertex_std @ extrinsic_righthand.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes[:4], facecolors=color, linewidths=0.3, edgecolors=color, alpha=1))
        self.ax.add_collection3d(
            Poly3DCollection(meshes[-1:], facecolors='g', linewidths=0.3, edgecolors=color, alpha=1))
        if name is not None:
            self.ax.text(vertex_transformed[0][0], vertex_transformed[0][1], vertex_transformed[0][2], name)

    def add_pointcloud(self, verts):
        # 3d scanner point cloud in left-hand
        self.ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.5)

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()
    
    def save(self, fn):
        self.fig.savefig(fn)

        

