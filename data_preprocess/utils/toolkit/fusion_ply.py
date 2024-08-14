import sys
sys.path.append('.')
import os
import numpy as np
from utils.logging import logging
from plyfile import PlyData, PlyElement

class PCDFusionToolkit:

    def __init__(self, target_vehicle_pcd_fn, scene_pcd_fn, scene_names, source_vehicle_name, save_dir):
        self._target_vehicle_pcd = PlyData.read(target_vehicle_pcd_fn)
        self._scene_pcd = PlyData.read(scene_pcd_fn)
        self._scene_names = scene_names
        self._source_vehicle_name = source_vehicle_name
        self._dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
        self._dtype_names = [i[0] for i in self._dtypes]
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)
        self._save_fn = os.path.join(self._save_dir, 'fusion.ply')
        self.run()
    
    def run(self):
        source_vehicle_pcd = None
        fused_pcd = []
        fused_attributes = []
        for element in self._scene_pcd.elements:
            logging.info(f'element name={element.name}, properties={len([e.name for e in element.properties])}')
            if element.name not in self._scene_names:
                if element.name == self._source_vehicle_name:
                    source_vehicle_pcd = np.vstack([element[e] for e in self._dtype_names]).T
                continue
            fused_attributes.append(np.vstack([element[e] for e in self._dtype_names]).T)

        if source_vehicle_pcd is None:
            logging.error(f'source vehicle is None!')
            raise ValueError

        target_vehicle_attributes = np.vstack([self._target_vehicle_pcd['vertex'][e] for e in self._dtype_names]).T
        fused_attributes = np.concatenate(fused_attributes + [target_vehicle_attributes], axis=0)

        fused_elements = np.empty(fused_attributes.shape[0], dtype=self._dtypes)
        fused_elements[:] = list(map(tuple, fused_attributes))

        fused_pcd.append(PlyElement.describe(fused_elements, 'vertex'))
        fused_ply = PlyData(fused_pcd)
        fused_ply.write(self._save_fn)


class PCDSIBRFusionToolkit:

    def __init__(self, target_vehicle_pcd_fn, scene_pcd_fn, scene_names, source_vehicle_name, save_dir):
        self._target_vehicle_pcd = PlyData.read(target_vehicle_pcd_fn)
        self._scene_pcd = PlyData.read(scene_pcd_fn)
        self._scene_names = scene_names
        self._source_vehicle_name = source_vehicle_name
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)
        self._save_fn = os.path.join(self._save_dir, 'fusion.ply')
        self.run()
    
    def run(self):
        fused_pcd = []
        dtypes = [(e.name, e.dtype()[1:]) for e in self._target_vehicle_pcd['vertex'].properties]
        total_attributes = np.vstack([self._target_vehicle_pcd['vertex'][e] for e, _ in dtypes]).T
        for element in self._scene_pcd.elements:
            logging.info(f'element name={element.name}, properties={len([e.name for e in element.properties])}')
            if element.name not in self._scene_names:
                continue
            dtypes = [(e.name, e.dtype()[1:]) for e in element.properties]
            for i in range(len(dtypes)):
                dtypes[i] = (dtypes[i][0], dtypes[i][1])
            attributes = np.vstack([element[e] for e, _ in dtypes]).T
            total_attributes = np.concatenate([total_attributes, attributes], axis=0)
        out_elements = np.empty(total_attributes.shape[0], dtype=dtypes)
        out_elements[:] = list(map(tuple, total_attributes))
        fused_pcd.append(PlyElement.describe(out_elements, element.name))
        fused_ply = PlyData(fused_pcd)
        fused_ply.write(self._save_fn)



