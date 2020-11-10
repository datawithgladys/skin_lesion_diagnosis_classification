# Capstone Project: Skin Lesion Classification and Diagnosis

### Gladys Pao, SG-DSI-17

## Problem Statement
Skin cancer is the most common cancer globally, with melanoma being the most deadly form. Even though dermoscopy, a skin imaging modality, has demonstrated improvement for the diagnosis of skin cancer compared to unaided visual inspection, numerous cases of benign lesions are still being diagnosed as malicious and vice versa. Every year, poor diagnostic errors adds an estimated $673 million in overall cost to manage the disease.

In this project, we aim to improve the diagnosis rate of skin cancer through the classification of skin lesions for dermatologists working at hospitals or skin clinics in Singapore, who will need experience or expertise in diagnosing skin cancer before they can accurately identify and diagnose lesions upon visual and dermoscopy inspection. This will be done through the classification of skin lesion dermoscopy images, in which we will predict two important tasks through the usage of Convolutional Neural Network models: <br>
1. a specific skin lesion diagnosis, and <br>
2. whether the lesion is malignant, benign, or pre-cancerous. <br>

The model will be evaluated based on its accuracy, followed by its recall rate since we are looking to minimise false negatives. Ultimately, we aim to get as close to a real evaluation of a dermatologist as possible: predicting the type of skin lesion; and whether the lesion is malignant, pre-cancerous or benign from dermoscopy images. With our models, we hope to aid dermatologists in their decision-making process of diagnosing skin lesions, hence allowing them to improve their diagnostic accuracy and come up with appropriate treatments for patients with skin lesions and/or cancers.

## Executive Summary
Skin cancer is the most common cancer globally, with melanoma being the most deadly form. Even though dermoscopy, a skin imaging modality, has demonstrated improvement for the diagnosis of skin cancer compared to unaided visual inspection, many benign lesions are still being misdiagnosed as malicious and vice versa. Every year, these misdiagnosis errors adds an estimated $673 million in overall cost to manage the disease. 

In this project, we aim to improve the diagnosis rate of skin cancer for dermatologists working at various hospitals and skin clinics in Singapore, who typically require a lot of experience before they can accurately identify and diagnose skin cancer upon visual inspection and dermoscopy. Through the usage of Convolutional Neural Network models, we will execute two important image classification tasks:
1. Classifying dermoscopic images based on their specific skin lesion, followed by
2. Classifying dermoscopic images based on their appropriate diagnosis (malignant, benign, or pre-cancerous)

To train our model, our data was collected from 6 data sources; subsets of images from 3 main data sources (ISIC 2018, 2019 and 2020 datasets) collected and combined to form our initial dermoscopy image dataset, before adding images from 3 external data sources (DermNetNZ, Dermscopedia and 7-Point Criteria Evaluation datasets) to tackle the problem of imbalanced classes and increase the number of images for the minority classes. In total, our final image dataset contains 10276 images from these 6 data sources. 
Models from both classification tasks will be mainly evaluated based on two main metrics, first on their accuracy rate, followed by their recall rate since we are looking to minimise false negatives.

With a moderately high accuracy of 70.53% for our skin lesion classification task and 76.36% for our diagnosis classification task, the overall performance of our CNN models for both classification tasks are moderately successful. In terms of their recall rates, both models have surpassed their baseline score for their respective tasks, with weighted averaged recall rates all exceeding 70% (70.53% for skin lesion; 76.36% for diagnosis) and macro-averaged recall rates all surpassing 60% (60.68% for skin lesion; 64.43% for diagnosis). With both accuracy and recall rates being lower for most of the minority classes, adding more image data for the minority classes should be of utmost priority when training the model in the future. 

### Contents:
- 1 Exploratory Data Analysis
- 2 Image Preprocessing
- 3a SMOTE for Skin Lesion Classifier
- 3b SMOTE for Diagnosis Classifier
- 4a Lesion Modelling and Evaluation
- 4b Diagnosis Modelling and Evaluation with Conclusion

## Overview of Data
The following links contain the sources of data for this project:<br>
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
|<div style="width:80px">Files</div>|<div style="width:100px">Notebook</div>|File Path|<div style="width:550px">Location/Links</div>|
|:---|:---:|:---|:---|
|All .csv files|1, 2, 4a, 4b|in their respective folders in /dataset|In this github repo|
|All .h5 files|4a, 4b|in their respective folders in /model|In this github repo|
|All .hdf5 files|4a, 4b|in their respective folders in /weights|In this github repo|
|Original images from image_dataset folder|1, 2|/datasets/image_dataset|https://drive.google.com/file/d/18aGpWmo3BDlK3fbRD4i5oLDzLAIbm53C/view?usp=sharing|
|Preprocessed images from processed_image folder|2, 3a, 3b|/datasets/processed_image|https://drive.google.com/file/d/1OTrpBH5KpNgw5gjBE4SI7PS6xa19o3N7/view?usp=sharing|
|X_test_lesion_284.npy, X_val_lesion_284.npy|4a|/datasets/npy/lesion/file.npy|https://drive.google.com/drive/folders/1OQkvKDiSHYP90keb8cVTWOG4bk0xDHjz?usp=sharing|
|X_smote_lesion_284.npy|4a|/datasets/npy/lesion/file.npy|https://mega.nz/file/FF1SHAoC#_oOQI7dsHFni0IG5gOmqmGQ9dUqjgH6z2nLqJSNXEx8|
|y_test_lesion_284.npy, y_val_lesion_284.npy, y_smote_lesion_284.npy|4a|/datasets/npy/lesion/file.npy|In this github repo|
|X_test_diagnosis_284.npy, X_val_diagnosis_284.npy|4b|/datasets/npy/diagnosis/file.npy|https://drive.google.com/drive/folders/1T1yhTehcU40l21JO3XnyTyDXunwfP9OA?usp=sharing|
|X_smote_diagnosis_284.npy|4b|/datasets/npy/diagnosis/file.npy|https://drive.google.com/file/d/1depWfTbfmBo1Xb8RhL4tsTJebYNL66_l/view?usp=sharing|
|y_test_diagnosis_284.npy, y_val_diagnosis_284.npy, y_diagnosis_lesion_284.npy|4b|/datasets/npy/diagnosis/file.npy|In this github repo|

## Data Dictionary
|<div style="width:70px">Skin Lesion</div>|<div style="width:70px">Abbreviation</div>|Diagnosis|<div style="width:120px">Description</div>|<div style="width:380px">Shape and Colour Chacteristics</div>|
|:---|:---|:---|:---|:------|
|**Actinic Keratoses**|AKIEC|Pre-cancerous|Includes Solar Keratoses and Bowen’s disease, which may progress to the  squamous cell carcinoma|Neoplasms commonly show surface scaling and commonly are devoid of pigment, though pigmented variants are still included in this set
|**Basal Cell Carcinoma**|BCC|Malignant|Common variant of epithelial skin cancer that rarely metastasizes but grows destructively if untreated.|Appears in different morphologic variants (flat, nodular, pigmented, cystic)|
|**Benign Keratosis-like Lesions**|BKL|Benign|Generic class that includes seborrheic keratoses/SK (senile wart), solar lentigo/SL (a flat variant of SK) and lichen-planus like keratoses/LPLK (SK or SL with inflammation and regression)|All three subgroups look different dermatoscopically; LPLK can show morphologic features mimicking MEL|
|**Dermatofibroma**|DF|Benign|A benign skin lesion regarded as either a benign proliferation or an inflammatory reaction to minimal trauma|Reticular lines at the periphery with a central white patch denoting fibrosis|
|**Melanoma**|MEL|Malignant|Malignant neoplasm derived from melanocytes; can be invasive or noninvasive|May appear in different variants and are usually, but not always, chaotic; dataset included all variants, except non-pigmented, subungual, ocular or mucosal melanoma
|**Melanocytic Nevi**|NV|Benign|Benign neoplasms of melanocytes|Appear in a myriad of variants, which are all included in this set; Usually more symmetric with regard to the distribution of color and structure|
|**Vascular Lesions**|VASC|Benign|Range from cherry angiomas and angiokeratomas to pyogenic granulomas and hemorrhage|Characterized by red or purple color and solid, well circumscribed structures known as red clods or lacunes|

## Conclusion and Recommendations
### Recommendations
Moving forward, the following recommendations can be considered:
1. With most misclassified images coming from minority classes, more data should be collected for those minority classes so that the model could learn more about the various types of colours, shapes, outlines and even size of the lesions from all these extra data collected. Even though SMOTE was used in this project to upsample the minority classes, using real images of the lesions from our minority classes would be much more useful for the model than synthetic samples created from SMOTE. <br>

2. Build a web application around our model so that the application could be used in-house securely at hospitals and skin clinics. Besides allowing dermatologists to make predictions with the web app, data could be also be added through this web app to further calibrate and improve our model (as explained above in the 1st point).<br>

3. Additional metadata/medical data of patients (such as age, gender) as well as characteristics of lesions as noted down by the physicians (e.g. where the skin lesion is located; shape and colour of lesions) could be added to our model for more detailed analysis and predictions.<br>

4. The classification tasks of both models can be expanded to include other types of skin lesions, such as non-neoplastic lesions for diagnosis classification, as well as cystic lesions and more variants of melanoma lesions for skin lesion classification.<br>

### Conclusion
With a moderately high accuracy of 70.53% for our skin lesion classification task and 76.36% for our diagnosis classification task, the overall performance of our CNN models for both classification tasks are moderately successful. Through there are still some room for improvement in their macro-averaged recall rates, both models have certainly surpassed the baseline score for their respective tasks, with macro-averaged recall rates all surpassing 60% (60.68% for skin lesion; 64.43% for diagnosis) and weighted averaged recall rates all exceeding 70% (70.53% for skin lesion; 76.36% for diagnosis). <br>
With both accuracy and recall rates being lower for the minority classes, adding more image data for the minority classes should be of utmost priority when training the model in the future. 

However, the models performed very well in terms of their accuracy, recall and precision rates for most benign skin lesions, especially for the diagnosis classification task, and hence could potentially serve as great classification models that can help improve the diagnosis rate specifically for benign skin lesions in the future.

Moving forward, the recommendations as stated above should be considered, which will not only calibrate the model for better performance through the usage of more data, but possibly also expand the model to train on other type of lesions so as to increase the model's use cases. By doing so, this model can serve as a better guide for dermatologists, aiding them in their decision-making process for patients with skin lesions or cancer, while continuously improving the clinical diagnostic accuracy of these skin specialists in the future.
