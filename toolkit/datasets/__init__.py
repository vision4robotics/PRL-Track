from .uav import UAVDataset
from .dtb import DTBDataset
from .uav10fps import UAV10Dataset
from .uav20l import UAV20Dataset
from .uav112l import UAV112LDataset
from .uav112 import UAV112Dataset
from .uavdt import UAVDTDataset
from .visdrone import VISDRONEDDataset


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert "name" in kwargs, "should provide dataset name"
        name = kwargs["name"]
        if "DTB70" in name:
            dataset = DTBDataset(**kwargs)
        elif "UAV123_10fps" in name:
            dataset = UAV10Dataset(**kwargs)
        elif "UAV123_20L" in name:
            dataset = UAV20Dataset(**kwargs)
        elif "UAV123" in name:
            dataset = UAVDataset(**kwargs)
        elif "UAVTrack112_L" in name:
            dataset = UAV112LDataset(**kwargs)
        elif "UAVTrack112" in name:
            dataset = UAV112Dataset(**kwargs)
        elif "UAVDT" in name:
            dataset = UAVDTDataset(**kwargs)
        elif "VISDRONED" in name:
            dataset = VISDRONEDDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs["name"]))
        return dataset
