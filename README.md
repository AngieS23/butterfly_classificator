# **Costa Rican Butterfly Classificator**

The following project contains a ConvNet architecture implemented in `pytorch` to classify the following four types of butterflies found in Costa Rica:

1. Siproeta Stelenes
2. Morpho Helenor
3. Euptoieta Hegesia Meridania
4. Biblis Hyperia Aganisa

## Dependencies

To use the code, certain python packages are required. You can install them by running the following command:

```
pip install -r requirements.txt
```

## Instruction Manual

### Directories

The project contains the following directories when you clone it.

```
butterfly_classificator/
├── results/
│   ├── epoch_metrics.csv
│   ├── final_results.csv
│   ├── testing_matrix.csv
│   └── training_matrix.csv
├── src/
│   ├── convnet.py
│   ├── csv_writer.py
│   └── trainer.py
└── utils/
    ├── data_augmenter.py
    ├── preprocessor.py
    └── splitter.py
```

1. In `results` you will find `.csv` files with the results from the last training execution. These can be used to plot the data.
2. In `src` you will find the relevant files to train the ConvNet model.
2. In `utils` you will find all files related to preprocessing the images.

To run any file with code you will need a version of the dataset.

### Dataset

The dataset for the project was constructed from photos taken from [Inaturalist](https://www.inaturalist.org/), you can download three datasets that were created for this project:

TODO(Luis): Add links later

1. [Initial dataset]() which contains the photos without being split.
2. [Split dataset]() which contains the photos split into a training, testing and validation split with a 80-10-10 proportion.
3. [Balanced dataset]() which contains a balanced version of the dataset through data augmentation.

### Auxiliary tools

If you want to create your own split, you can use the auxiliary tools in the `utils` directory to:
1. Resize the images.
2. Split the data.
3. Balance the classes.

#### Resizing the images

The images were resized to 200x200 pixels to be easier to process by the model, to do this you must have a `data` directory with the images and execute the following command:

```
python preprocessor.py
```

This will create a new directory titled `preprocessed_data`.

#### Splitting the data

To split the data you must have the `preprocessed_data` directory from the previous step and execute the following command:

```
python splitter.py
```

This will create a new directory titled `split_data`with the following splits:
1. Training: 80% of the images.
2. Testing: 10% of the images.
3. Validation: 10% of the images.

If you want to modify these splits you can change the following line in the code:

```
splitfolders.ratio(input_path, output=output_path, seed=18, ratio=(0.80, 0.10, 0.10))
```

### Training the ConvNet

Once you either downloaded the dataset or created your own version you can train the model by running the following command inside the `src` directory:

```
python trainer.py
```

This will update the `.csv` files inside the `results` directory and will output the following statements in the command line:

```
    TODO(Luis): Add prints
```

### Result visualization

TODO(Luis): Add link to the colab


## Made by
- [Angie Solís Manzano](https://github.com/AngieS23)
- [Luis David Solano Santamaría](https://github.com/GoninDS) 