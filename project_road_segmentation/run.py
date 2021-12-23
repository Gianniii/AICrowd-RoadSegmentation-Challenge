import os
from scipy.ndimage import rotate
import re
import torch
import random
from models import *
from train import *
from torch import nn
from torch import optim
from helper import *
from datetime import datetime
from numpy import moveaxis
from evaluation import *
import statistics
import time
import sys

if "--load" in sys.argv:
    LOAD_MODEL = True
    i = sys.argv.index("--load")
    CHECKPOINT_PATH = sys.argv[i+1]
    print("Using checkpoint... No training")
else:
    LOAD_MODEL = False
    CHECKPOINT_PATH = None

# Hyperparameters etc.
PATCH_SIZE = 16
TRAIN_IMG_DIR = "../data/training/images/"
TRAIN_MASK_DIR = "../data/training/groundtruth/"
TEST_IMG_DIR = "../data/test_set_images/"
#LEARNING_RATE = [1e-5, 1e-4, 1e-3, 1e-2]
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 400  # original size of train images
IMAGE_WIDTH = 400   # original size of train images
BATCH_SIZE = 4
NUM_EPOCHS = 10 # ideal number of epochs is the maximum until the model converges
AUGMENTATION = True  # Do image augmentations
SUBMISSION = True # Output submission with provided test set

SEED = 66478  # Set to None for random seed.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


files = os.listdir(TRAIN_IMG_DIR)
random.shuffle(files)
nbr_train = min(80, len(files))  # nbr train samples
# nbr of test samples / nbr_train + nbr_test should not exceed len(files)
nbr_test = min(19, len(files))

# move axes to change image to channel first format
print("Loading " + str(nbr_test + nbr_train) + " images")

test_eval_gt_imgs = [load_image(TRAIN_MASK_DIR + files[i])
                     for i in range(nbr_train, nbr_train + nbr_test)]

# This is to sort correctly the directory
def sorted_alphanumeric(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


# Load the submission images 
if SUBMISSION:
    test_files = os.listdir(TEST_IMG_DIR)
    test_files = sorted_alphanumeric(test_files)
    test_imgs_SUB = [moveaxis(load_image(TEST_IMG_DIR + test_files[i - 1] +
                            "/test_" + "{:d}".format(i) + ".png",), 2, 0,) for i in range(1, 51)]

# =======================  DATA AUGMENTATION  ======================= #

# SCIPY BUG: add this line
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if AUGMENTATION:
    train_imgs = []
    for i in range(nbr_train):
        loaded_img = load_image(TRAIN_IMG_DIR + files[i])
        train_imgs.append(moveaxis(loaded_img, 2, 0))
        # Image rotation
        train_imgs.append(moveaxis(rotate(loaded_img, 90), 2, 0))
        train_imgs.append(moveaxis(rotate(loaded_img, 2 * 90), 2, 0))
        #train_imgs.append(moveaxis(rotate(loaded_img, 3 * 90), 2, 0))
        #train_imgs.append(moveaxis(rotate(loaded_img, 1 * 15), 2, 0, reshape=False))
        #train_imgs.append(moveaxis(rotate(loaded_img, 2 * 15), 2, 0, reshape=False))
        train_imgs.append(moveaxis(rotate(loaded_img, 3 * 15, reshape=False), 2, 0)) # 45 degrees
        # Image mirrored
        #train_imgs.append(moveaxis(loaded_img[:, ::-1, :], 2, 0)) 

    train_gt_imgs = []
    for i in range(nbr_train):
        loaded_img = load_image(TRAIN_MASK_DIR + files[i])
        train_gt_imgs.append(loaded_img)
        # Image rotation
        train_gt_imgs.append(rotate(loaded_img, 90))
        train_gt_imgs.append(rotate(loaded_img, 2 * 90))
        #train_gt_imgs.append(rotate(loaded_img, 3 * 90))
        #train_gt_imgs.append(rotate(loaded_img, 1 * 15, reshape=False))
        #train_gt_imgs.append(rotate(loaded_img, 2 * 15, reshape=False))
        train_gt_imgs.append(rotate(loaded_img, 3 * 15, reshape=False)) # 45 degrees
        # Image mirrored
        #train_gt_imgs.append(loaded_img[:, ::-1])
else:
    # ret[nbr_test_imgs, 3 ,400, 400]
    train_imgs = [moveaxis(load_image(TRAIN_IMG_DIR + files[i]), 2, 0)
                  for i in range(nbr_train)]
    train_gt_imgs = [load_image(TRAIN_MASK_DIR + files[i])
                     for i in range(nbr_train)]

# =======================  DATA LOADING  ======================= #

# Load test images with ground their respective ground truths
test_imgs = [moveaxis(load_image(TRAIN_IMG_DIR + files[i]), 2, 0)
             for i in range(nbr_train, nbr_train + nbr_test)]
test_gt_imgs = [load_image(TRAIN_MASK_DIR + files[i])
                for i in range(nbr_train, nbr_train + nbr_test)]

# images for evaluation MUST NOT BE RESIZED
test_eval_imgs_og = [load_image(TRAIN_IMG_DIR + files[i])
                     for i in range(nbr_train, nbr_train + nbr_test)]
test_eval_gt_imgs = [load_image(TRAIN_MASK_DIR + files[i])
                     for i in range(nbr_train, nbr_train + nbr_test)]

# Get patches for evaluation
test_gt_patches = [img_crop( test_eval_gt_imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(nbr_test)]
# Linearize list of patches
test_gt_patches = np.asarray([test_gt_patches[i][j] for i in range(
    len(test_gt_patches)) for j in range(len(test_gt_patches[i]))])

# Variables for logging
runtimes = []
error_rate_logs = []
error_rate_patches = []
f1score_logs = []
f1_patches_logs = []
nbr_runs = 1

# =======================  MAIN  ======================= #

for i in range(nbr_runs):
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    train_input = torch.FloatTensor(np.array(train_imgs))  
    test_input = torch.FloatTensor(np.array(test_imgs))
    train_target = (torch.FloatTensor(np.array(train_gt_imgs)) > 0.5).float()
    test_target = (torch.FloatTensor(np.array(test_gt_imgs)) > 0.5).float()

    if SUBMISSION:
        test_input = torch.FloatTensor(np.array(test_imgs_SUB))
    if LOAD_MODEL: 
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
    else:
        # Train model
        start_time = time.time()
        train_model(model, train_input, train_target, LEARNING_RATE,
            BATCH_SIZE, NUM_EPOCHS, DEVICE, opt="Adam") 
        runtime = time.time() - start_time
        runtimes.append(runtime)
        print("Run " + str(i + 1) + ": Training time: " + str(runtime))
    


    # Evaluate model on test set
    if SUBMISSION:
        create_submission_images(model, test_input)
    else:
        # evaluate model
        nbr_errors, f1_score, patches_nbr_errors, patches_f1_score = evaluate_unet_model(
            model, test_input, test_target, test_eval_gt_imgs, test_eval_imgs_og, test_gt_patches, DEVICE)
        

    # for lr cross-validation
    """
    for lr in LEARNING_RATE:
        start_time = time.time()
        train_model(model, train_input, train_target, lr, BATCH_SIZE,
            NUM_EPOCHS, DEVICE, opt="Adam", topo_gamma=0.0)  
        runtime = time.time() - start_time
        runtimes.append(runtime)
        print("Run " + str(i + 1) + ": Training time: " + str(runtime))

        # Evaluate model on test set
        nbr_errors, f1_score, patches_nbr_errors, patches_f1_score = evaluate_unet_model(
            model, test_input, test_target, test_eval_gt_imgs, test_eval_imgs_og, test_gt_patches,
            (IMAGE_HEIGHT, IMAGE_WIDTH), RESIZE_BOOL, SUBMISSION) """
    
    if not SUBMISSION:
        # print results
        rate = nbr_errors / (IMAGE_HEIGHT * IMAGE_WIDTH * nbr_test)
        num_patches_per_image = (IMAGE_HEIGHT / PATCH_SIZE) * \
            (IMAGE_WIDTH / PATCH_SIZE)
        rate_patches = patches_nbr_errors  / (num_patches_per_image * nbr_test)

        #print(f"Test errors:\t\t\t{nbr_errors:.2e}\nTest error rate:\t\t{rate:.2f}")
        #print(f"F1 Score: \t\t\t{f1_score:.2f}")
        print(f"Patches Test errors:\t\t{patches_nbr_errors:.2e}\nPatches Test error rate:\t{rate_patches:.2f}")
        print(f"Patches F1 Score:\t\t{patches_f1_score:.2f}")
        print("==================\n")
        
        # Logging
        error_rate_logs.append(rate)
        error_rate_patches.append(rate_patches)
        f1score_logs.append(f1_score)
        f1_patches_logs.append(patches_f1_score)
        
    if not LOAD_MODEL:
        # save trained model
        print("\n==================")
        print("Saving model ...")
        model_checkpoint_directory = "model_checkpoints/"
        if not os.path.isdir(model_checkpoint_directory):
            print("Create directory\n")
            os.mkdir(model_checkpoint_directory)
        model_path = ("model_checkpoints/model_" +
                    datetime.now().strftime("%d_%H_%M_%S") + ".checkpoint")
        torch.save(model.state_dict(), model_path)
        print(f"Saved as: %s" % model_path)

    # For the next run clear the cache if using cuda
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print("\n==================")

#print("Writing outputs ...\n")

# Print results
if nbr_runs > 1:
    error_rate_logs_mean = np.array(error_rate_logs).mean()
    error_rate_logs_std = np.array(error_rate_logs).std()

    error_rate_patches_mean = np.array(error_rate_patches).mean()
    error_rate_patches_std = np.array(error_rate_patches).std()

    print(f"Statistics after %d runs" % nbr_runs)
    print(f"Average training time:\t\t\t%f sec" % statistics.mean(runtimes))
    
    print(f"Mean of test error rates:\t\t{error_rate_logs_mean:.2f} ± {error_rate_logs_std:.2f}")   
    print(f"Mean of F1 scores:\t\t\t%f" % statistics.mean(f1score_logs))
    print(f"Mean of test error rates for patches:\t{error_rate_patches_mean:.2f} ± {error_rate_patches_std:.2f}")   
    print(f"Mean of F1 scores for patches:\t\t%f" % statistics.mean(f1_patches_logs))
    print("==================\n")
    
#print("Writing outputs ...\n")
