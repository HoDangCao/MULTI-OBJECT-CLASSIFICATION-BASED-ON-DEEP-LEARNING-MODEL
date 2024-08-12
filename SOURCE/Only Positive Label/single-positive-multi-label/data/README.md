# Getting the Data
## COCO

1. Navigate to the COCO data directory:
```
cd /path/to/single-positive/multi-label/data/coco
```
2. Download the data:
```
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip --output coco_annotations.zip
curl http://images.cocodataset.org/zips/train2014.zip --output coco_train_raw.zip
curl http://images.cocodataset.org/zips/val2014.zip --output coco_val_raw.zip
```
3. Extract the data:
```
unzip -q coco_annotations.zip
unzip -q coco_train_raw.zip
unzip -q coco_val_raw.zip
```
4. Clean up:
```
rm coco_train_raw.zip
rm coco_val_raw.zip
```
5. Download the pre-extracted features for COCO from [here](https://zenodo.org/records/10162606) and copy them to `/path/to/single-positive/multi-label/data/coco`.
# Formatting the Data
The `preproc` folder contains a few scripts which can be used to produce uniformly formatted image lists and labels:
```
cd /path/to/single-positive/multi-label/preproc
python format_coco.py
```
# Generating Observed Labels
The script `preproc/generate_observed_labels.py` subsamples the entries of a complete label matrix to generate "observed labels" which simulate single positive labeling. To generate observed labels for a given dataset, run:
```
cd /path/to/single-positive/multi-label/preproc
python generate_observed_labels.py --dataset `coco`