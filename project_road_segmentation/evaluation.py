import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
from sklearn.metrics import f1_score
import torchvision
from torchvision import transforms
from PIL import Image
from numpy import moveaxis
from mask_to_submission import *
from helper import *

# input and targets are a lists of patches
# counts errors and writes predictions in an output file
def evaluate_unet_model(model, inputs, targets, gt_imgs, test_imgs, test_gt_patches, device, PATCH_SIZE=16, batch_size = 4):
    """Evaluates the produced network on the test set

    Args:
        model (torch.nn.Module): A torch neural network
        inputs (torch 4d tensor): A tensor of shape (num_images, num_channels, width, height)
        targets (torch 4d tensor): A tensor of shape (num_images, 1, width, height)
        gt_imgs: The ground truth as PIL Images
        test_imgs: The inputs as PIL Images
        test_gt_patches: The ground truth as PIL Images, in patches
        device (str): cuda or cpu
        PATCH_SIZE (int, optional): The patch size. Defaults to 16.
        batch_size (int, optional): The batch size. Defaults to 4.

    Returns:
        (int, float, int, float): 
            The total number of errors,
            the F1 score,
            the total number of errors on the patches,
            the F1 score on the patches.
    """
    with torch.no_grad():
        total_num_errors = 0
        predicted_patches = []
        total_predictions = []
        for b in range(0, len(inputs), batch_size):
            if b + batch_size < len(inputs):
                b_max = b + batch_size
                batch_resize = batch_size
            else:
                b_max = len(inputs)
                batch_resize = len(inputs) - b

            batch_inputs = inputs[b: b_max].to(device)
            batch_targets= targets[b: b_max].to(device)
            outputs = model(batch_inputs)
            for cur_image_index in range(0, batch_resize):
                output = outputs[cur_image_index]

                pred = torch.sigmoid(output)
                predicted_im = (pred > 0.5).float().cpu().numpy().squeeze()

                predicted_patches.append(predicted_im)

                new_p = Image.fromarray(np.uint8(predicted_im * 255), "L")

                # # write prediction as an image to an output file
                prediction_training_dir = "predictions_training/"
                if not os.path.isdir(prediction_training_dir):
                    print("Create directory\n")
                    os.mkdir(prediction_training_dir)
            
                # Concatenate ground truth and prediction
                gt_pil = Image.fromarray(np.uint8(gt_imgs[cur_image_index + b] * 255), "L")
                concat_sat_gt = concatenate_images(test_imgs[cur_image_index + b], np.array(gt_pil))
                cimg = concatenate_images(concat_sat_gt, np.array(new_p))

                print(f"Saving image %d" % (b + cur_image_index + 1))
                new_p = Image.fromarray(cimg)
                new_p.save(prediction_training_dir + "prediction_" +
                        str(cur_image_index + b) + ".png")

                overlay_img = make_img_overlay(
                    test_imgs[cur_image_index + b], predicted_im)
                overlay_img.save(prediction_training_dir +
                                "overlay_" + str(cur_image_index + b) + ".png")

            batch_predictions = (torch.sigmoid(outputs) > 0.5).float()
            for i in range(batch_predictions.shape[0]):
                total_predictions.append(batch_predictions[i])

            batch_num_errors = (batch_predictions != batch_targets).view(-1).sum()
            total_num_errors += batch_num_errors

        total_predictions = torch.stack(total_predictions)
        # get predictions of patches
        predicted_patches = [img_crop(
            predicted_patches[i], PATCH_SIZE, PATCH_SIZE)for i in range(len(predicted_patches))]

        predicted_patches = np.asarray([predicted_patches[i][j] for i in range(
            len(predicted_patches)) for j in range(len(predicted_patches[i]))])

        # assign label to patches of predicted and gt
        groundtruth_patch_labels = torch.tensor([value_to_class(
            np.mean(test_gt_patches[i])) for i in range(len(test_gt_patches))])
        predicted_patch_labels = torch.tensor([value_to_class(
            np.mean(predicted_patches[i])) for i in range(len(predicted_patches))])

        # compare these labels and return f1 score and total num error for patches
        num_patch_errors = (predicted_patch_labels != groundtruth_patch_labels).sum()

    return (total_num_errors.item(),
            f1_score(total_predictions.cpu().flatten(),targets.cpu().flatten()),
            num_patch_errors.item(),
            f1_score(predicted_patch_labels.flatten(),groundtruth_patch_labels.flatten()),
    )


def create_submission_images(model, input):
    model.to("cpu")
   
    for cur_image_index in range(len(input)):
        image = input.narrow(dim=0, start=cur_image_index, length=1) 
        
        output_image = model(image)
        pred = torch.sigmoid(output_image)
        predicted_im = (pred > 0.5).float().cpu().numpy().squeeze()

        print(f"Saving image %d" % (cur_image_index + 1))
        new_p = Image.fromarray(np.uint8(predicted_im * 255), "L")
        new_p.save("../data/submission/sub_" +
                    str(cur_image_index + 1) + ".png")

    submission_filename = "submission.csv"
    image_filenames = []
    for i in range(1, 51):
        # image_filename = 'training/groundtruth/satImage_' + '%.3d' % i + '.png'
        image_filename = "../data/submission/sub_" + "{:d}".format(i) + ".png"
        print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)    
    
    return 



def evaluate_FFN_model(model, input, target, gt_imgs, patch_size, test_imgs):
    count = 0  # counts number of errors
    for cur_image_index in range(len(test_imgs)):
        j = cur_image_index * num_patches_per_image
        output = model(
            input[j: j + num_patches_per_image, :, :, :]
        )  # 625 patches in first image of test set !!
        _, predicted_classes = output.max(1)

        # count number of errors
        for k in range(input[j: j + num_patches_per_image, :, :, :].size(0)):
            if target[k] != predicted_classes[k]:
                count = count + 1

        # write prediction as an image to an output file

        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            print("Create directory\n")
            os.mkdir(prediction_training_dir)

        w = gt_imgs[cur_image_index].shape[0]
        h = gt_imgs[cur_image_index].shape[1]
        predicted_im = label_to_img(
            w, h, patch_size, patch_size, predicted_classes)
        cimg = concatenate_images(test_imgs[cur_image_index], predicted_im)
        Image.fromarray(cimg).save(
            prediction_training_dir + "prediction_" +
            str(cur_image_index) + ".png"
        )
        overlay_img = make_img_overlay(
            test_imgs[cur_image_index], predicted_im)
        overlay_img.save(
            prediction_training_dir + "overlay_" +
            str(cur_image_index) + ".png"
        )

    return count
