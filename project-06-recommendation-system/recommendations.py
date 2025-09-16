# System for recommending similar products based on image likeness using a Neural Network




#%%gg
## Download dataset
from genericpath import exists
from re import I
import kagglehub
import os
import shutil

from sympy import O


path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

dest = os.path.join(os.getcwd(), "dataset")

if not os.path.exists(dest):
    shutil.copytree(path, dest)
    
print(f"Dataset copied to: {dest}")


# %%
## Prepare Data

# separate every image to it's own category folder
# by checking the styles.csv file

import pandas as pd
from shutil import move
import os
from tqdm import tqdm

curr_dir = os.getcwd()

# ensure the data categories folder is created
os.makedirs("fashion_data/categories", exist_ok=True)
categories_path = os.path.join(curr_dir, "fashion_data", "categories")

# dataset path
dataset_path = os.path.join(curr_dir, "dataset")
dataset_images_path = os.path.join(dataset_path, "images")



df = pd.read_csv(
    os.path.join(dataset_path, "styles.csv"),
    usecols=["id", "masterCategory"]
    ).reset_index()

df["id"] = df["id"].astype("str")

all_images = os.listdir(dataset_images_path)
cnt = 0

for img in tqdm(all_images):
    category = df[df["id"] == img.split(".")[0]]["masterCategory"]
    category = str(list(category)[0])
    os.makedirs(os.path.join(categories_path, category), exist_ok=True)

    path_from = os.path.join(dataset_images_path, img)
    path_to = os.path.join(categories_path, category, img)
    move(path_from, path_to)
    cnt += 1
    
print(f"Moved {cnt} images.")



# %% [markdown]
## Train a model using Transfer Learning 

#%%
# import the necessary modules and check if gpu is available for tensorflow

import itertools
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
from torchvision import transforms



os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras_hub as khub
import torch

print("Keras version:", keras.__version__)
print("Torch version:", torch.__version__)
print("Hub version:", khub.__version__)
print("Keras backend", keras.backend.backend())
print("GPU is available" if torch.cuda.is_available() else "GPU is not available")

#%%
# define the variables and parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32 
N_FEATURES = 256

data_dir = os.path.join(os.getcwd(), "fashion_data", "categories")

#%%
# create the validation and train datasets


# create validation dataset
valid_dataset = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.20,
    subset="validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# create training dataset
train_dataset = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.20,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

qty_train_classes = len(train_dataset.class_names) # type: ignore


normalization_layer = keras.layers.Rescaling(1./255)

# apply normalization to validation dataset
valid_dataset = valid_dataset.map(lambda x, y: (normalization_layer(x), y)) # type: ignore
valid_batches = valid_dataset.cardinality().numpy() # type: ignore
approx_valid_samples = valid_batches * BATCH_SIZE

do_data_augmentation = False
if do_data_augmentation:
    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomTranslation(0.2, 0.2),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomShear(0.2)
    ])
    
    train_batches = train_dataset.cardinality().numpy()   # type: ignore
    approx_train_samples = train_batches * BATCH_SIZE
    
    train_dataset = train_dataset.map(                              # type: ignore
        lambda x, y: (data_augmentation(normalization_layer(x)), y)
    )
else:
    train_batches = train_dataset.cardinality().numpy()  # type: ignore
    approx_train_samples = train_batches * BATCH_SIZE
    
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y)) # type: ignore



#%%
import keras

# buiding the model
print(qty_train_classes)

backbone = khub.models.ResNetBackbone.from_preset("resnet_v2_101_imagenet")
backbone.trainable = False

model = keras.Sequential([
    backbone,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(N_FEATURES, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(qty_train_classes, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.0001)),    

], name="resnet_transfer_model")

# model.build((None,)+IMAGE_SIZE+(3,))
model.summary()


#%% 
# define optmizer and loss
# lr = 0.003 * BATCH_SIZE / 512 
# SCHEDULE_LENGTH = 500
# SCHEDULE_BOUNDARIES = [200, 300, 400]


# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
# lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES, values=[lr, lr*0.1, lr*0.001, lr*0.0001])
# optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

# loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), # type: ignore
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

#%%
## train model
# steps_per_epoch = approx_train_samples // BATCH_SIZE
# validation_steps = approx_valid_samples // BATCH_SIZE
# hist = model.fit(
#     train_dataset,
#     epochs=5,
#     steps_per_epoch=steps_per_epoch,
#     validation_data=valid_dataset,
#     validation_steps=validation_steps).history

#%%
# view graphs
# plt.figure()
# plt.ylabel("Loss (training and validation)")
# plt.xlabel("Training Steps")
# plt.ylim([0,2])
# plt.plot(hist["loss"])
# plt.plot(hist["val_loss"])

# plt.figure()
# plt.ylabel("Accuracy (training and validation)")
# plt.xlabel("Training Steps")
# plt.ylim([0,1])
# plt.plot(hist["accuracy"])
# plt.plot(hist["val_accuracy"])




#%%
# save model and feature_extractor

model_path = os.path.join(os.getcwd(), "model", "model.keras")
model.save(model_path)

feature_extractor_path = os.path.join(os.getcwd(), "model", "feature_extractor", "feature_extractor.keras")
feature_extractor = keras.Model(inputs=model.inputs, outputs=model.layers[-3].output)
feature_extractor.save(feature_extractor_path)

#%% reload the model
model_path = os.path.join(os.getcwd(), "model", "model.keras")
feature_extractor_path = os.path.join(os.getcwd(), "model", 'feature_extractor', 'feature_extractor.keras')

model = keras.models.load_model(model_path)
feature_extractor = keras.models.load_model(feature_extractor_path)

#%%
# function to load the images

def load_image(path):
    img = Image.open(path).convert("RGB")  # ensure 3 channels
    transform = transforms.Compose([
        transforms.Resize(224),             # resize smallest side to 224
        transforms.CenterCrop(224),         # crop center
        transforms.ToTensor(),              # convert to [0, 1] float tensor
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)    # type: ignore | add batch dimension
    img_tensor = img_tensor.permute(0, 2, 3, 1) # shape [1, 224, 224, 3]
    
    return img_tensor                      
#%%
# vectorize the images

# get a list of randomized paths to images
tqdm.pandas()

path_to_categories = os.path.join(os.getcwd(), "fashion_data", "categories")

img_paths = []

for path in Path(path_to_categories).rglob("*.jpg"):
    img_paths.append(path)
np.random.shuffle(img_paths)



imgvec_path = os.path.join(os.getcwd(), "image_vectors")
os.makedirs(imgvec_path, exist_ok=True)


for filename in tqdm(img_paths[:5000]):
    img = load_image(str(filename))
    features = feature_extractor(img)   # type: ignore
    feature_set = features.detach().squeeze().cpu().numpy()
    outfile_name = os.path.basename(filename).split('.')[0] + '.csv'
    out_path_file = os.path.join(imgvec_path, outfile_name)
    np.savetxt(out_path_file, feature_set, delimiter=',')
    
# %%
## metadata and indexing
#hide
import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import json
from annoy import AnnoyIndex
from scipy import spatial
import pickle
from IPython.display import Image as dispImage

dataset_styles_path = os.path.join(os.getcwd(), "dataset", "styles.csv")
root_styles_path = os.path.join(os.getcwd(), "styles.csv")
image_vectors_path = os.path.join(os.getcwd(), "image_vectors", "*.csv")

# get test image
test_img = os.path.join(os.getcwd(), "fashion_data", "categories", "Accessories", "1941.jpg")
dispImage(test_img)

# read the styles.csv file

styles = pd.read_csv(dataset_styles_path, on_bad_lines='skip')
styles['id'] = styles['id'].astype('str')
styles.to_csv(root_styles_path, index=False)


def match_id(fname: str, styles: pd.DataFrame):
    return styles.index[styles.id==fname].values[0]


# Defining data structures as empty dict
file_index_to_file_name = {}
file_index_to_file_vector = {}
file_index_to_product_id = {}

# configure annoy parameters
dims = 256
n_nearest_neighbors = 20
trees = 10_000

all_files = glob.glob(image_vectors_path)

t = AnnoyIndex(dims, metric="angular")

for findex, fname in tqdm(enumerate(all_files)):
    file_vector = np.loadtxt(fname, delimiter=',')
    file_name = os.path.basename(fname).split('.')[0]
    file_index_to_file_name[findex] = file_name
    file_index_to_file_vector[findex] = file_vector
    try:
        file_index_to_product_id[findex] = match_id(file_name, styles)
    except IndexError:
        pass
    t.add_item(findex, file_vector)


t.build(trees)
t.save('t.ann')

vector_data_path = os.path.join(os.getcwd(), "vector_data")

t.save(os.path.join(vector_data_path, "indexer.ann"))
pickle.dump(file_index_to_file_name, open(os.path.join(vector_data_path, "file_index_to_file_name.p"), "wb"))
pickle.dump(file_index_to_file_vector, open(os.path.join(vector_data_path, "file_index_to_file_vector.p"), "wb"))
pickle.dump(file_index_to_product_id, open(os.path.join(vector_data_path, "file_index_to_product_id.p"), "wb"))




# %%
## Use model to suggest similar images
from PIL import Image
import matplotlib.image as mpim

topK = 5

path_to_categories = os.path.join(os.getcwd(), "fashion_data", "categories")
test_image_path = os.path.join(os.getcwd(), "test_purse_image.jpg")

test_vec = np.squeeze(feature_extractor(load_image(test_image_path)))   # type: ignore

basewidth = 224
img = Image.open(test_image_path)
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)


path_dict = {}
for path in Path(path_to_categories).rglob('*.jpg'):
    path_dict[path.name] = path

nns = t.get_nns_by_vector(test_vec, n=topK)

plt.figure(figsize=(20, 10))

for i in range(topK):
    x = file_index_to_file_name[nns[i]]
    x = path_dict[x+'.jpg']
    y = file_index_to_product_id[nns[i]]
    title = '\n'.join([str(j) for j in list(styles.loc[y].values[-5:])])
    plt.subplot(1, topK, i+1)
    plt.title(title)
    plt.imshow(mpim.imread(x))
    plt.axis('off')
plt.tight_layout()
