# Landmark Classification & Tagging for Social Media

End-to-end CNN project for the Udacity "Convolutional Neural Networks" course.
Classifies images of landmarks into 50 classes using (a) a CNN trained from
scratch and (b) a transfer-learned ResNet-50.

## Layout

```
landmark_classifier/
├── Project_Landmarks_Part1_CNNfromScratch__starter.ipynb
├── Project_Landmarks_Part2_TransferLearning__starter.ipynb
├── Project_Landmarks_Part3_App__starter.ipynb
├── src/
│   ├── data.py          # ImageFolder + transforms + DataLoaders
│   ├── model.py         # MyModel — from-scratch CNN
│   ├── transfer.py      # get_model_transfer_learning(...)
│   ├── optimization.py  # get_loss / get_optimizer
│   ├── train.py         # train_one_epoch / valid_one_epoch / optimize / one_epoch_test
│   ├── predictor.py     # Predictor — TorchScript-scriptable inference wrapper
│   └── helpers.py       # data download, mean/std, plotting
├── checkpoints/         # written by training
├── static_images/       # drop a test image here for Part 3
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook Project_Landmarks_Part1_CNNfromScratch__starter.ipynb
```

The dataset is downloaded automatically by `src.helpers.setup_env()` if it is
not already present in the working directory.

## Running the tests

Each module has its own pytest suite. Run them with:

```bash
pytest -vv src/
```

The tests load a small subset (`limit=200`) of the data so they run in seconds
once the dataset is available.

## Expected accuracy

| model              | test accuracy |
| ------------------ | ------------- |
| from-scratch CNN   | ≥ 50% (target), ~60% with tuning |
| transfer (ResNet50)| ≥ 60% (target), ~80%+ with tuning |
