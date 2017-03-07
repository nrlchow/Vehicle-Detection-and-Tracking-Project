# Vehicle Detection and Tacking Project

In this project, my goal is to write a software pipeline to detect vehicles in a video.


The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run a pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



[//]: # (Image References)
[image1]: ./output_images/car_notcar_image.png "data_exploration"
[image2]: ./output_images/bin_spatial.png "bin_spatial"
[image3]: ./output_images/Hog_Visualisation.png "Hog_Visualisation"
[image4]: ./output_images/extract_single_image_feature.png "extract_single_image_feature"
[image5]: ./output_images/Normalized_Features_image.png "Normalized_Feature_image"
[image6]: ./output_images/slide_window_96x96.png "window_search"
[image7]: ./output_images/slide_window_128x128.png "window_search2"
[image8]: ./output_images/multiple_detections1.png "multiple_detections1"
[image9]: ./output_images/multiple_detections2.png "multiple_detections2"
[image10]: ./output_images/multiple-detections-removed-heatmap.png "heatmap"
[image11]: ./output_images/Heatmap-Final-box.png  "Heatmap-Final-box"
[image12]: ./output_images/pipeline0.png "pipeline0"
[image13]: ./output_images/pipeline1.png "pipeline1"
[image14]: ./output_images/pipeline2.png "pipeline2" 
[image15]: ./output_images/pipeline3.png "pipeline3"
[image16]: ./output_images/pipeline4.png "pipeline4"

Video Reference:

### Project video

[![Alt text](https://img.youtube.com/vi/IJd4m20nfzE/0.jpg)](https://www.youtube.com/watch?v=IJd4m20nfzE)


## Data Exploration
I have started by exploring the 'vehicle' and 'non-vehicle' images.The datasets are comprised of images taken from the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself.
The downloaded data consists of 8792 car images and 8968 non-car images.The shape of the image is (64, 64, 3), and the data type of the image is float32. 
I have written 'data_look()' function to extract this information. 
The code for exploring and visualizing the data is contained in the "Data Exploration" and "Data Visualisation "  cells of the IPython notebook.

Shown below is an example of each class (vehicle, non-vehicle) of the data set.

![alt text][image1]


## Extract Features

I have extracted features using color_hist(),bin_spatial(),get_hog_features functions. 

### Histograms of Color

I have written 'color_hist()' function to compute the histogram of the color channels separately. 
The color histogram is a representation of the distribution of colors in an image. It is produced first by discretization of the colors in the image into a number of bins, 
and then by counting the number of image pixels in each bin.

In the 'color_hist()' function, I split the image into three channels, and then I got each histogram.'np.histogram()' returned a tuple of two arrays.
'channel1_hist[0]' contained the counts in each of the bins and 'channel1_hist[1]' contained the bin edges.I settled with histogram bin size of 32 after repeated experiments. 

The code is contained in the "Color Histogram Features"  cell of the IPython notebook. 


### Spatial Binning of Color

I have written 'bin_spatial() function to compute spatial binned color features. This function converts test image into a feature vector. 
This function uses raw pixel values to include in feature vector.They are quite useful in searching for cars.  
The function uses cv2.resize().ravel() to create the feature vector. I have settled with (64,64) spatial binning dimensions which increased 
the feature length to 17676 and seemed very useful during training the classifier.

The code is contained in the "Binned Color Features"  cell of the IPython notebook. 
 
Which gives us this result:

![alt text][image2]

### HOG Features

I have written 'get_hog_features()' function to give us HOG features and HOG image.
The histogram of oriented gradients(HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. 
The technique counts occurrences of gradient orientation in localized portions of an image.

The 'scikit-image' package has a built in function to extract Histogram of Oriented Gradient features.Using this built in function, I defined a 'get_hog_features()' function to return HOG features and image. 
I explored different color spaces and different 'skimage.hog()' parameters which are 'orient', 'pix_per_cell', and 'cell_per_block'.
I performed repated experiments on random images from each of the two classes and displayed them to get a feel for what the 'skimage.hog()' output looks like. 
I combined the hog features from all the colour channels in the YCrCb color space, which seemed to work fine.
I used HOG parameters of orientations equal to 9, (8, 8) pixels_per_cell and (2, 2) cells_per_block.
I exracted features on all channels as it gives better result.

Car image and its corresponding HOG visulization, they look like this:

![alt text][image3]

The code for extracting the HOG features is contained in the "Histogram of Oriented Gradients (HOG)" cell of the IPython notebook.


### Combine and Normalize Features

I have written a function that extracts feature vector from image by combining the three techniques shown above. 
'extract_features()' function takes in a list of car and not-car image separately,reads them one by one, then applies a color conversion and uses 'bin_spatial()
and 'color_hist()' and 'get_hog_features()' each of which generates  1-dementional feature vectors.They are stacked ontop of each other for a final feature vector representation.

I have also written 'single_img_features()' function to extract 1-D feature vector from any given single image by combining the same feature extraction technique shown above. 
The code for extracting the feature vectors is contrained in the "Extract Features from a List of Images" and "Single Image Features Extract" cells of the IPython notebook.
 
I almost ready to train a classifier, but I need to normalize my data. 'sklearn' package provides me with the 'StandardScaler()' method to accomplish this task.

The result of extracting the feature vector from the image and normalizing it is as follows:


# Extract single image feature
![alt text][image4]

# Normalized Feature image
![alt text][image5]


## Train a classifier
To train a classifier on the extracted features vector,I defined a labels vector.I have split up data into randomized training and test sets. 
Using train_test_split()' function of `sklearn` package, I have used 10% of scaled data for data validaition and testing purpose.
Finally, defined a classifier and trained it. 

I have written 'classifier()' function that can take any of these classifiers i.e LinearSVC(),RandomForestClassifier(),KNeighborsClassifier(),keras model as input and train model using user's choice of classifier. 
It then trains and saves the model using pickles.It also returns validation/test score. 
I experimented with many different combination of parameters and classifiers to find how the model performs. 
I found a combination of parameters that yield the best results after repeated experiments and used that to train the model. 
I found LinearSVC() and RandomForestClassifier() yielded similar validation score.Finally,I settled with linear SVM.
The SVM has achieved classification test accuracy of 99.32% with a test set that is 10% of the orginal training data. 
This high accuracy was very important for the detection pipeline in order to minimize the number of false postive and negative detections.

The code for this function is contained in the 'Classifier function' cell of the IPython notebook. 
 
## Sliding Window and Search - Classify
Now it's time to search for cars and I have all the tools for it. I have implemented 'slide_window()' function to detect a set of windows where vehciles and actual road is likey to be in the image or video frame.
This list of windows are later fed to 'search_windows()' to be searched for that iterates over all windows in the list.While doing iteration,
the function extracts features for that window using 'single_img_features()' function and performs prediction using the trained classifier.If its a positive prediction, the window gets saved.

I have implemented the sliding window approach at 2 different sizes:128x128,96x96. These size has worked well to detect cars with bounded boxes. 
I have used an overlap of 0.5,05 and settled with 'x start stop' of [630,1280]  and 'y start stop' of [390,600] positions that has seemed worked best.


The results are as follows:

# Window search 96x96
![alt text][image6]

# Window search 128x128
![alt text][image7]

##  Multiple Detections and False Positives

# Multiple_detections1
![alt text][image8]

# Multiple_detections2
![alt text][image9]

I have found the classifier reported multiple overlapping detections for each of the vehicles after applying bounding boxes.
I have built heat-map from these detections in order to combine overlapping detections and remove false positives. 
To do this, I have written two functions, 'add_heat ()' and 'apply_threshold ()' and applied threshold. 

The 'test_bounding_boxes()' function code is contained in the 'Multiple Detections and Bounding Box Test'  cell of the IPython notebook. 
The heat-map function codes are contained in  'Heat-map and False Positives' cell of the IPython notebook. 
The heat-map visualization code is contained in 'Visualiaze heatmap & Save Hot-Windows' cell of the IPython notebook. 

Finally,I have used the 'label()' function from 'scikit-image' and writtente a 'draw_labeled_bboxes()' function to find final boxes from heatmap and put bounding boxes around the labeled regions. 

The following images show this process.

# Multiple detections removed using heatmap
![alt text][image10]

# Heatmap and Final-box
![alt text][image11]


## Pipeline
I have built a 'video_pipeline()' function that combines all the work so far. This function detects a car by inputting a single image and returns an image showing the position of the car as a box. I tested with images in the 'test_images directory.

# pipeline0
![alt text][image12]

# pipeline1
![alt text][image13]

# pipeline2
![alt text][image14]

# pipeline3
![alt text][image15]

# pipeline4
![alt text][image16]


I have applied the pipeline to the both test and project video and confirmed they performed as expected. 
This completed video can be found [here]

### Project video

[video1]: ./project_video_1.mp4 "project video"

### Test Video
[video2]: ./test_video_1.mp4 "Test Video"

## Discussion
This project was very challenging as it required a lot of parameters tuining.At times,I found it was hard to isolate the problem area due to use of overwhelming number of parameters that requires manual tuning.
This can be improved by implementing a dynamic function like "grid search" that can provide a best combination of parameters for given set of test images.
Alternatively,an iterative process can be implemeted by feeding a list of values as input but that will require a lot of computationally efficient hardwares.  
The pipeline will likey pick a lot of unwanted vehicles that are travellig from opposite direction or other side of roads if they are made visible.
While training the classifier,I primarily focused on getting the SVM classifier to produce the best results.The classifier reported good results.


