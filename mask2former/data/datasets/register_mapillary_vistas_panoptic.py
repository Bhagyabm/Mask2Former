# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


MAPILLARY_VISTAS_SEM_SEG_CATEGORIES = [
 {
            "id": 1,
            "name": "Static Obstacle",
            "supercategory": "obstacle",
            "isthing": 0,
            "color": [
                255,
                212,
                25
            ]
        },
        {
            "id": 2,
            "name": "Water",
            "supercategory": "water",
            "isthing": 0,
            "color": [
                70,
                245,
                255
            ]
        },
        {
            "id": 3,
            "name": "Sky",
            "supercategory": "sky",
            "isthing": 0,
            "color": [
                170,
                0,
                255
            ]
        },
        {
            "id": 4,
            "name": "Boat/ship",
            "supercategory": "obstacle",
            "isthing": 1,
            "color": [
                255,
                39,
                43
            ]
        },
        {
            "id": 5,
            "name": "Row boats",
            "supercategory": "obstacle",
            "isthing": 1,
            "color": [
                255,
                39,
                43
            ]
        },
        {
            "id": 6,
            "name": "Paddle board",
            "supercategory": "obstacle",
            "isthing": 1,
            "color": [
                255,
                39,
                43
            ]
        },
        {
            "id": 7,
            "name": "Buoy",
            "supercategory": "obstacle",
            "isthing": 1,
            "color": [
                255,
                39,
                43
            ]
        },
        {
            "id": 8,
            "name": "Swimmer",
            "supercategory": "obstacle",
            "isthing": 1,
            "color": [
                255,
                39,
                43
            ]
        },
        {
            "id": 9,
            "name": "Animal",
            "supercategory": "obstacle",
            "isthing": 1,
            "color": [
                255,
                39,
                43
            ]
        },
        {
            "id": 10,
            "name": "Float",
            "supercategory": "obstacle",
            "isthing": 1,
            "color": [
                255,
                39,
                43
            ]
        },
        {
            "id": 11,
            "name": "Other",
            "supercategory": "obstacle",
            "isthing": 1,
            "color": [
                255,
                39,
                43
            ]
        }
]


def load_mapillary_vistas_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        Using Custom Datasets </tutorials/datasets.html>_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_mapillary_vistas_panoptic(
    name, metadata, image_root, panoptic_root, semantic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named name.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            register_coco_panoptic_separated.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_mapillary_vistas_panoptic_json(
            panoptic_json, image_root, panoptic_root, semantic_root, metadata
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="mapillary_vistas_panoptic_seg",
        ignore_label=255,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_ADE20K_PANOPTIC = {
    "mapillary_vistas_panoptic_train": (
        "/content/drive/My Drive/Lars/train/images",
        "/content/drive/My Drive/Lars/train/panoptic_masks",
        "/content/drive/My Drive/Lars/train/path_to_updated_train_file.json",
        "/content/drive/My Drive/Lars/train/semantic_masks",
    ),
    "mapillary_vistas_panoptic_val": (
        "/content/drive/My Drive/Lars/val/images",
        "/content/drive/My Drive/Lars/val/panoptic_masks",
        "/content/drive/My Drive/Lars/val/path_to_updated_val_file.json",
        "/content/drive/My Drive/Lars/val/semantic_masks",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_" and "stuff_" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    thing_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    stuff_classes = [k["name"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]
    stuff_colors = [k["color"] for k in MAPILLARY_VISTAS_SEM_SEG_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(MAPILLARY_VISTAS_SEM_SEG_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_mapillary_vistas_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_ADE20K_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_mapillary_vistas_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_mapillary_vistas_panoptic(_root)
