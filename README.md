# Capstone Project: Skin Lesion Classification and Diagnosis

### Gladys Pao, SG-DSI-17

## Problem Statement
Skin cancer is the most common cancer globally, with melanoma being the most deadly form. Even though dermoscopy, a skin imaging modality, has demonstrated improvement for the diagnosis of skin cancer compared to unaided visual inspection, numerous cases of benign lesions are still being diagnosed as malicious and vice versa. Every year, poor diagnostic errors adds an estimated $673 million in overall cost to manage the disease.

In this project, we aim to improve the diagnostic rate of skin cancer through the classification of skin lesions for dermatologists working at hospitals or skin cancer clinics in Singapore, who will need experience or expertise in diagnosing skin cancer before they can accurately identify and diagnose lesions upon visual and dermoscopy inspection. This will be done through the classification of skin lesion dermoscopy images, in which we will predict two important tasks through the usage of Convolutional Neural Network models: <br>
1. a specific skin lesion diagnosis, and <br>
2. whether the lesion is malignant, benign, or pre-cancerous. <br>

The model will be evaluated based on its accuracy, followed by its recall rate since we are looking to minimise false negatives. Ultimately, we aim to get as close to a real evaluation of a dermatologist as possible: predicting the type of skin lesion; and whether the lesion is malignant, pre-cancerous or benign from dermoscopy images. With our models, we hope to aid dermatologists in their decision-making process of diagnosing skin lesions, hence allowing them to improve their diagnostic accuracy and come up with appropriate treatments for patients with skin lesions and/or cancers.


## Executive Summary
Skin cancer is the most common cancer globally, with melanoma being the most deadly form. Even though dermoscopy, a skin imaging modality, has demonstrated improvement for the diagnosis of skin cancer compared to unaided visual inspection, many benign lesions are still being misdiagnosed as malicious and vice versa. Every year, these misdiagnosis errors adds an estimated $673 million in overall cost to manage the disease. 

In this project, we aim to improve the diagnostic rate of skin cancer for dermatologists working at various hospitals and skin cancer clinics in Singapore, who typically require a lot of experience before they can accurately identify and diagnose skin cancer upon visual inspection and dermoscopy. Through the usage of Convolutional Neural Network models, we will execute two important image classification tasks:
1. Classifying dermoscopic images based on their specific skin lesion, followed by
2. Classifying dermoscopic images based on their appropriate diagnosis (malignant, benign, or pre-cancerous)

To train our model, our data was collected from 6 data sources; subsets of images from 3 main data sources (ISIC 2018, 2019 and 2020 datasets) collected and combined to form our initial dermoscopy image dataset, before adding images from 3 external data sources (DermNetNZ, Dermscopedia and 7-Point Criteria Evaluation datasets) to tackle the problem of imbalanced classes and increase the number of images for the minority classes. In total, our final image dataset contains 10276 images from these 6 data sources. 

Resizing, central cropping, padding, and colour normalisation (white balancing grey-world algorithm) were techniques used during image preprocessing to standardise the images in terms of their colours and sizes. Morphological filtering and inpainting methods were also utilised to remove black hairs and markings present in numerous images. Synthetic Minority Oversampling Technique (SMOTE) was also used to generate synthetic samples for the minority classes to obtain a class-balanced training set.

For modelling and evaluation, our CNN models from both classification tasks will be mainly evaluated based on two main metrics, first on their accuracy rate, followed by their recall rate since we are looking to minimise false negatives. With a moderately high accuracy of 70.53% for our skin lesion classification task and 76.36% for our diagnosis classification task, the overall performance of our CNN models for both classification tasks are moderately successful. In terms of their recall rates, both models have surpassed their baseline score for their respective tasks, with weighted averaged recall rates all exceeding 70% (70.53% for skin lesion; 76.36% for diagnosis) and macro-averaged recall rates all surpassing 60% (60.68% for skin lesion; 64.43% for diagnosis). With both accuracy and recall rates being lower for most of the minority classes, adding more image data for the minority classes should be of utmost priority when training the model in the future. 

### Contents:
- 1 - Exploratory Data Analysis
- 2 - Image Preprocessing
- 3a - SMOTE for Skin Lesion Classifier
- 3b - SMOTE for Diagnosis Classifier
- 4a - Lesion Modelling and Evaluation
- 4b - Diagnosis Modelling and Evaluation with Conclusion


## Overview of Data
<b>Main Data Sources:</b><br>
International Skin Imaging Collaboration (ISIC) 2018, 2019 and 2020 datasets:<br>
- BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona<br>
- HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161<br>
- MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368<br>
- The ISIC 2020 Challenge Dataset: (c) by ISDIS; https://doi.org/10.34970/2020-ds01<br>

<b>Other Data Sources:</b>
- DermNet NZ: (c) New Zealand Dermatological Society; https://dermnetnz.org/<br>
- Dermoscopedia: (c) Dermoscopedia; https://dermoscopedia.org/<br>
- 7-point Criteria evaluation Database: (c) by Jeremy Kawahara, Sara Daneshvar, Giuseppe Argenziano, and Ghassan Hamarneh; https://doi.org/10.1109/JBHI.2018.2824327


### Location of Data
The following files can be found at their respective locations or download links:

|<div style="width:60px">Files</div>|Notebook|File Path|Location/Links|
|:---|:---:|:---|:---|
|All .csv, .h5, .hdf5 files|1, 2, 4a, 4b|in their respective folders|In this github repo|
|Original images from image_dataset|1, 2|/datasets/|https://bit.ly/image_dataset|
|Preprocessed images from processed_image folder|2, 3a, 3b|/datasets/|https://bit.ly/processed_image|
|X_test_lesion_284.npy, X_val_lesion_284.npy|4a|/datasets/npy/lesion/|https://bit.ly/x-test-x-val-lesion|
|X_smote_lesion_284.npy|4a|/datasets/npy/lesion/|https://bit.ly/x-smote-lesion|
|y_test_lesion_284.npy, y_val_lesion_284.npy, y_smote_lesion_284.npy|4a|/datasets/npy/lesion/|In this github repo|
|X_test_diagnosis_284.npy, X_val_diagnosis_284.npy|4b|/datasets/npy/diagnosis/|https://bit.ly/x-test-x-val-diagnosis|
|X_smote_diagnosis_284.npy|4b|/datasets/npy/diagnosis/|https://bit.ly/x-smote-diagnosis|
|y_test_diagnosis_284.npy, y_val_diagnosis_284.npy, y_smote_diagnosis_284.npy|4b|/datasets/npy/diagnosis/|In this github repo|


## Data Dictionary
Skin Lesion|Abbreviation|Diagnosis|
|:---|:---|:---|
|**Actinic Keratoses**|AKIEC|Pre-cancerous|
|**Basal Cell Carcinoma**|BCC|Malignant|
|**Benign Keratosis-like Lesions**|BKL|Benign|
|**Dermatofibroma**|DF|Benign|
|**Melanoma**|MEL|Malignant|
|**Melanocytic Nevi**|NV|Benign|
|**Vascular Lesions**|VASC|Benign|


## Conclusion and Recommendations
### Recommendations
Moving forward, the following recommendations can be considered:
1. With most misclassified images coming from minority classes, more data should be collected for those minority classes so that the model could learn more about the various types of colours, shapes, outlines and even size of the lesions from all these extra data collected. Even though SMOTE was used in this project to upsample the minority classes, using real images of the lesions from our minority classes would be much more useful for the model than synthetic samples created from SMOTE. <br>

2. Build a web application around our model so that the application could be used in-house securely at hospitals and skin cancer clinics. Besides allowing dermatologists to make predictions with the web app, data could be also be added through this web app to further calibrate and improve our model (as explained above in the 1st point).<br>

3. Additional metadata/medical data of patients (e.g. age, gender) or characteristics of lesions as noted down by the physicians (e.g. location of lesion) could be feeded to a fully connected network before merging the outputs of both sub-models (dense neural network and convolutional neural network models) for more detailed predictions and analysis.<br>

4. The classification tasks of both models can be expanded to include other types of skin lesions, such as non-neoplastic lesions for diagnosis classification, as well as cystic lesions and other variants of melanoma lesions for skin lesion classification.<br>

### Conclusion
With a moderately high accuracy of 70.53% for our skin lesion classification task and 76.36% for our diagnosis classification task, the overall performance of our CNN models for both classification tasks are moderately successful. Through there are still some room for improvement in their macro-averaged recall rates, both models have certainly surpassed the baseline score for their respective tasks, with macro-averaged recall rates all surpassing 60% (60.68% for skin lesion; 64.43% for diagnosis) and weighted averaged recall rates all exceeding 70% (70.53% for skin lesion; 76.36% for diagnosis). <br>
With both accuracy and recall rates being lower for the minority classes, adding more image data for the minority classes should be of utmost priority when training the model in the future. 

However, the models performed very well in terms of their accuracy, recall and precision rates for most benign skin lesions, especially for the diagnosis classification task, and hence could potentially serve as great classification models that can help improve the diagnosis rate specifically for benign skin lesions in the future.

Moving forward, the recommendations as stated above should be considered, which will not only calibrate the model for better performance through the usage of more data, but possibly also expand the model to train on other type of lesions so as to increase the model's use cases. By doing so, this model can serve as a better guide for dermatologists, aiding them in their decision-making process for patients with skin lesions or cancer, while continuously improving the clinical diagnostic accuracy of these specialists in the future.
