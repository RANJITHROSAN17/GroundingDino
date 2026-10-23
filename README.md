# Grounding DINO auto-labelling

Zero-shot **automatic bounding-box labelling** for custom object-detection datasets, built on
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and
[supervision](https://github.com/roboflow/supervision).

Give it a folder of images and a list of class names in plain English — it detects every instance,
lets you eyeball the results, and exports **Pascal VOC XML annotations** ready to be refined in a
labelling tool or used to train a conventional detector (YOLO, Faster R-CNN, …). No training, no
pre-existing labels.

Built for a PPE / industrial-safety use case (`person`, `helmet`, `vibrocharger`), but the classes
are just strings — change them and it works for anything Grounding DINO can describe.

## How it works

```
images/ ──► Grounding DINO (Swin-T, open-vocabulary) ──► boxes + class ids + confidences
                       │  prompt: "all persons", "all helmets", …
                       ▼
           supervision Detections ──► preview grid ──► Pascal VOC XML (annotations/)
```

1. **Setup** – clones Grounding DINO at a pinned commit, installs it, downloads the
   `groundingdino_swint_ogc.pth` checkpoint.
2. **Single-image check** – runs one image with your classes and plots the annotated result so you
   can tune the thresholds before processing the whole set.
3. **Batch labelling** – iterates over every `jpg/jpeg/png` in `IMAGES_DIRECTORY`, keeps detections
   that resolved to one of your classes, and shows an *original vs. annotated* grid.
4. **Export** – writes one VOC XML per image via `sv.Dataset.as_pascal_voc(...)` and zips the
   `annotations/` folder.

## Running it

The pipeline lives in `grounding_dino_auto_label.py`, which is an exported **Google Colab notebook**
(it contains `!pip`, `!wget` and `drive.mount` cells), so the intended way to run it is to paste it
back into Colab with a GPU runtime:

1. Open a new Colab notebook → *Runtime → Change runtime type → GPU*.
2. Copy the contents of `grounding_dino_auto_label.py` into cells (each block separated by a blank
   line is one cell), or upload it via *File → Upload notebook* after renaming to `.ipynb`.
3. Put your images in a zip on Google Drive and adjust `SOURCE_IMAGE_PATH` / the `unzip` path.
4. Edit the parameters:

   ```python
   CLASSES        = ["person", "helmet"]   # anything describable in English
   BOX_TRESHOLD   = 0.35                   # min box confidence  (raise → fewer, surer boxes)
   TEXT_TRESHOLD  = 0.15                   # min text-match confidence
   ```

5. Run all cells. Download `annotations.zip` at the end.

Running locally instead of Colab: install the pinned Grounding DINO commit and
`requirements.txt`, remove the `!`-prefixed shell lines and the Drive mount, and run the script
with Python ≥ 3.9 on a CUDA machine (CPU works but is slow).

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && git checkout 57535c5a79791cb76e36fdb64975271354f10251 && pip install -e .
mkdir weights && wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
pip install -r ../requirements.txt
```

## Tips that made a difference

- Prompting with `"all helmets"` instead of `"helmet"` (`enhance_class_name`) noticeably improved
  recall for multiple instances.
- Filter `detections[detections.class_id != None]` — Grounding DINO sometimes returns boxes whose
  phrase does not map back to any of your classes.
- `MIN/MAX_IMAGE_AREA_PERCENTAGE` in the VOC export drop boxes that are suspiciously tiny or cover
  almost the whole image; both are usually false positives.
- Treat the output as a **first pass**: review in a labelling tool (CVAT, Label Studio, Roboflow)
  before training on it.

## Pinned versions

| component | version |
|---|---|
| GroundingDINO | commit `57535c5` (v0.1.0-alpha weights, Swin-T OGC) |
| supervision | 0.4.0 (the `Detections` tuple layout changed in later versions) |

## License

Code in this repository is released under the [Unlicense](LICENSE). Grounding DINO is Apache-2.0
and supervision is MIT; the model weights are subject to the Grounding DINO release terms.
