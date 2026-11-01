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

[**Open the runnable notebook in Colab**](https://colab.research.google.com/github/RANJITHROSAN17/GroundingDino/blob/main/grounding_dino_auto_label.ipynb) · [Inspect notebook source](grounding_dino_auto_label.ipynb)

**Status:** legacy GPU workflow with pinned dependencies. The notebook structure and directory setup are checked locally; GPU inference, downloads and export compatibility have not been revalidated here. Review generated boxes before using them for training.

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

Use `grounding_dino_auto_label.ipynb`, a real notebook with the exported Colab shell and directory magics restored. The `.py` export is not a standalone Python program, and renaming it to `.ipynb` does not convert it.

1. Open the linked notebook in Colab → *Runtime → Change runtime type → GPU*.
2. Inspect the setup and Drive access before running the pipeline cell. Installation and weights require network downloads; do not put sensitive images in an unfamiliar cloud runtime.
3. Put your images in a zip on Google Drive and adjust `SOURCE_IMAGE_PATH` / the `unzip` path.
4. Edit the parameters:

   ```python
   CLASSES        = ["person", "helmet"]   # anything describable in English
   BOX_TRESHOLD   = 0.35                   # min box confidence  (raise → fewer, surer boxes)
   TEXT_TRESHOLD  = 0.15                   # min text-match confidence
   ```

5. Run all cells. Download `annotations.zip` at the end.

### Local execution is a porting task

This repository does not currently provide a tested local CLI. A local port must replace Colab Drive paths, IPython magics and downloads, and reproduce the pinned Grounding DINO/supervision environment. Do not run the exported `.py` with plain Python and expect the Colab shell syntax to work.

### Check the notebook without downloading weights

```bash
python -m unittest discover -s tests -v
```

These standard-library checks verify notebook structure, a clean output state and directory setup. They do not verify GPU inference or the upstream dependencies. A useful issue includes the failing cell, traceback, GPU/runtime versions and pinned dependency versions, without private Drive URLs or images.

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

## Maintainer

**[Ranjith Rosan](https://github.com/RANJITHROSAN17)** maintains this repository’s Grounding DINO auto-labelling workflow. Grounding DINO and supervision are separate upstream projects; their authors and licenses retain their original credit.

## License

Code in this repository is released under the [Unlicense](LICENSE). Grounding DINO is Apache-2.0
and supervision is MIT; the model weights are subject to the Grounding DINO release terms.
