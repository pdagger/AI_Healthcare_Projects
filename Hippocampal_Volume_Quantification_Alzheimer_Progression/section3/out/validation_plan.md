# Validation plan

# Intended use:
For assisting radiologists in identifying and measuring hyppocampal volumes from T2 MRI studies.

# Training data:
The data comes from the "Hippocampus" dataset from the [Medical Decathlon competition](http://medicaldecathlon.com/). This dataset is stored as a collection of NIFTI files, with one file per volume, and one file per corresponding segmentation mask. The original images are T2 MRI scans of the full brain but in this dataset we are using cropped volumes where only the region around the hippocampus has been cut out. This makes the size of the dataset quite a bit smaller, the machine learning problem a bit simpler and allows to have reasonable training times. 

The data was used as follows:
- Training: 80%
- Validation: 10%
- Testing: 10%

# Labels of training data:
According to [medicaldecathlon.com](http://medicaldecathlon.com/): "All data has been labeled and verified by an expert human rater, and with the best effort to mimic the accuracy required for clinical use. For more information on the data, please refer to [https://arxiv.org/abs/1902.09063](https://arxiv.org/abs/1902.09063)"

# Training performance of the algorithm measured vs real-world performance going to be estimated:
The training performance of the algorithm is measured using the Jacard similarity coefficient and Dice score comparing the predicted volume to the training volume.

The real-world performance is going to be estimated by averaging the volume identification results from 3 experienced radiologists.

# What data will the algorithm perform well in the real world and what data it might not perform well on?
The algorithm will perform well in 3D T2 MRI scans of the brain from adult female and male subjects where each voxel has a cube shape.

The algorithm may not peform well in images with rectangular pixels or MRI scans other than of type T2.