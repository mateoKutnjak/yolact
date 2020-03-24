# Instructions

Steps:
1. Acquire Blender generated dataset from densefusion BitBucket project
2. python3 converters/blender2coco.py --src ../dataset --dest ../coco_dataset_550 --img-width 550 --img-height 550
3. mv ../coco_dataset_550/* data/coco/
4. mv data/coco/annotations/* data/coco
(OPTIONAL) python3 converters/COCOVisualize.py data/coco/train.json
5.a python3 train.py --config custom_yolact_plus_101_config --dataset dataset_custom --batch_size 3
5.b polyaxon run -u -f polyaxonfile.yml -P config=custom_yolact_plus_101_config_polyaxon -P batch_size=3
6. Model wights are now inside weights/ directory

