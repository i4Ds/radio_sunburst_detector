import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd


# Obtain training, validation and test datasets from dataframes
def get_datasets(burst_df, noburst_df):
    
    train_size = int(0.7 * len(burst_df))  #Same size for noburst_df since they're equal
    val_size = int(0.15 * len(burst_df))  #Same size for noburst_df since they're equal

    train_burst = burst_df[:train_size]
    val_burst = burst_df[train_size : train_size + val_size]
    test_burst = burst_df[train_size + val_size :]

    train_noburst = noburst_df[:train_size]
    val_noburst = noburst_df[train_size : train_size + val_size]
    test_noburst = noburst_df[train_size + val_size:]
    
    train_df = pd.concat([train_burst, train_noburst]).sample(frac=1).reset_index(drop=True)
    val_df = pd.concat([val_burst, val_noburst]).sample(frac=1).reset_index(drop=True)
    test_df = pd.concat([test_burst, test_noburst]).sample(frac=1).reset_index(drop=True)

    datagen = ImageDataGenerator(rescale=1./255.)
    directory = os.getcwd()
    
    train_ds = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=directory,  
        x_col="file_path",
        y_col="label",
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="grayscale"
    )
    val_ds = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=directory,  
        x_col="file_path",
        y_col="label",
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="grayscale"
    )    

    test_ds = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=directory,  
        x_col="file_path",
        y_col="label",
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="grayscale"
    )    
    

    return train_ds, val_ds, test_ds, train_df, val_df, test_df


if __name__ == "__main__":
    
    configured_burst_df = pd.read_excel('configured_burst.xlsx')
    configured_noburst_df = pd.read_excel('configured_noburst.xlsx')

    train_ds, validation_ds, test_ds, train_df, val_df, test_df = get_datasets(configured_burst_df, configured_noburst_df)


    class_names = list(train_ds.class_indices.keys())
    for ds, ds_name, df in zip([train_ds, validation_ds, test_ds], ['train', 'validation', 'test'], [train_df, val_df, test_df]):
            #Plot some images
            images, labels = next(ds)  #get the next batch of images and labels from the generator
            plt.figure(figsize=(10, 10))
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i], cmap="gray")
                plt.title(f"{class_names[int(labels[i])]} ({labels[i]}).")
                plt.axis("off")
            plt.show()

            # Calculate class balance
            print(f"Class balance in {ds_name} dataset:")
            print(df['label'].value_counts())
    
