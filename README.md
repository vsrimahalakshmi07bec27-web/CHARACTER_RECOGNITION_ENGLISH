Developed a CNN-based alphanumeric character recognition system with OpenCV-based segmentation, supporting 62 classes and automated image annotation.
# CHARACTER_RECOGNITION_ENGLISH
This project implements a Character Recognition System using a pre-trained Convolutional Neural Network (CNN) combined with image segmentation techniques.The system detects and recognizes individual alphanumeric characters (0–9, A–Z, a–z) from input images and generates annotated output with bounding boxes and predicted labels.
Key Features

-- Character segmentation using OpenCV contour detection

-- Preprocessing with grayscale conversion, Gaussian blur & Otsu thresholding

--Recognition using a trained CNN model (.h5)

--Supports 62 classes (0–9, A–Z, a–z)

-- Bounding box visualization with predicted labels

-- Batch processing of multiple test images

-- Automatic annotated output image saving
 TECH STACK:

--Python

--OpenCV

--NumPy

--TensorFlow / Keras

WORKING PIPLEINE:

Load trained CNN model

Preprocess image (grayscale → blur → threshold)

Detect contours for character segmentation

Sort characters left-to-right

Resize to 32×32 and normalize

Predict character using CNN

Draw bounding box and label

Save annotated output

OUTPUT:

Prints recognized text in console

Saves annotated images with bounding boxes

Processes all images in test directory

APPLICATIONS:

Optical Character Recognition (OCR)

Automated document digitization

License plate recognition (extendable)

Embedded vision systems
