# FDA  Submission

**Your Name:**

**Name of your Device:**

## Algorithm Description 

### 1. General Information

**Intended Use Statement:** 

For assisting the radiologist in the detection of pneumonia in a chest X-ray image.

**Indications for Use:**

Screening chest X-ray studies taken in PA or AP positions from any human subject between ages 2-90 years old. The human may present or not any other chest condition detectable by X-ray.

**Device Limitations:**

The algorithm is best adapted for subjects between 20-70 years old.

**Clinical Impact of Performance:**

The algotihm was trained in a huge dataset where only 1.28% of the images corresponded to patients with pneumonia. Then it is designed to have a high number of false positives in order to minimize false negatives.

### 2. Algorithm Design and Function

Model plot: "model_1: Model" corresponds to a pretrained VGG16 architecture.

![Model plot: "model_1: Model" corresponds to a pretrained VGG16 architecture.](./model_plot.png)

**DICOM Checking Steps:**

Images checked to be gray images.

**Preprocessing Steps:**

If images are RGB, changed to gray.
Images scaled bewteen 0 and 255 (divided by 255).

**CNN Architecture:**

CNN architecture VGG16 pretrained model.
VGG16 architecture up to pooling of 3rd convolutional layer from block 5.

<img src="./model_1_VGG16.png" alt="CNN architecture VGG16 pretrained model." width="600"/>

### 3. Algorithm Training

**Parameters:**
* Types of augmentation used during training:
  - Horizontal_flip.
  - Vertical_flip.
  - Height shift between 0 and 0.1.
  - Width shift between 0 and 0.1.
  - Rotation between 0° and 20°.
  - Shear between 0 and 0.1.
  - Zoom between 0 and 0.1

* Batch size:
  - Training: 16.
  - Validation: 6037.

* Optimizer learning rate: 0.0001.
* Layers of pre-existing architecture that were frozen:
first 16 layers.

* Layers of pre-existing architecture that were fine-tuned: 17 and 18 (see image above).
* Layers added to pre-existing architecture: 11 layers added (image below).

Model_1 below corresponds to pretrained VGG16 architecture.

<img src="./added_layers.png" alt="Added layers." width="600"/>

* Training performance visualization:

<img src="./performance_visual.png" alt="performance." width="600"/>

 * P-R curve:

<img src="./p-r_curve.png" alt="performance." width="600"/>

**Final Threshold and Explanation:**

Given that the classes of interest in the dataset are so unbalanced, the behavior of the non dominant class was considered and thus more importance was given to the recall metric. For the develped model, lower thresholds give rise to better recalls but they considerably increase the false positive rates. A threshold of 0.45 giving an acceptable recall of ~0.7 was choosen as it gives a good balance between true positves and false positives: it will detect true positives ~70% of the times without exagerating the account of false positives.

### 4. Databases
 (For the below, include visualizations as they are useful and relevant)

**Description of Training Dataset:** 


**Description of Validation Dataset:** 


### 5. Ground Truth



### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

**Ground Truth Acquisition Methodology:**

**Algorithm Performance Standard:**
