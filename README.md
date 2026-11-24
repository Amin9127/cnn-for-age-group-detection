# CNN for Age Group Detection

PyTorch coursework notebook for classifying face images into four age groups: Child, Young, Middle-Aged, and Senior.

`age_group_cnn.ipynb` loads images from `train/` and `test/` folders, resizes them to 128x128 grayscale, splits a validation set, and trains `AgeGroupCNN` with class weights (`compute_class_weight`) so the model does not collapse onto the majority class.

At the end it prints test accuracy, a classification report, and a confusion matrix.

Dependencies: Python 3, PyTorch, Torchvision, scikit-learn, Pillow, NumPy, Matplotlib.
