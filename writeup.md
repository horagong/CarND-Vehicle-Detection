## Writeup 
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 3rd ~ 8th code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `gray` color space and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters like cell_per_block, pix_per_cell, orient, hog_channel. Finally I chose orient=15 and cell_per_block=2, pix_per_cell=8, hog_channel='ALL', colorspace='YUV' considering accuray and computation speed.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using hog parameters above and color histgram of 16 but no spatial binning. I thought spatial binning had no big effect on the test video result because the cars had so many different colors.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search over the images with three different scaled window. I changed the window size with respect to the height position of the window like this image.

| Height    | scale |
|:---------:|:-----:|
| 380 ~ 500 | 1.0   |
| 400 ~ 650 | 1.4   |
| 450 ~ 680 | 2.5   |



![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on three scales using YUV 3-channel HOG features plus histograms of color in the feature vector. When I used 15 of orientation, it provided a nice result.  

I detected features from the test image using sliding window.

![alt text][image4]

After that, I add the count of detected boxs to a heatmap and give a threshold on it.

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In 13th code cell, I tried to use some methods to filter for false positives and combine overlapping bounding boxes.

I first used three heatmaps to give a different threshold to each scaled window  becase I thought the rate of false positives would be different according to window size. A small sized window had the higher rate of false positives I gave it the higher theshold value and it was good.

```python
bbox_list1 = threshold_on_heat(img, heats1, 7, bbox_list1) 
bbox_list2 = threshold_on_heat(img, heats2, 5, bbox_list2)
bbox_list3 = threshold_on_heat(img, heats3, 3, bbox_list3)
```

But it needed some more computation time. So I decided to use just one heatmap and adjust a threshold considering false positives between frames.

I accumulated five frames to a queue and applied thresold on sum of heats in the queue using threshold_on_heat() in the 11th code cell and then recorded the positions of positive detections in each frame of the video.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem was that it was not realtime to process to detect vehicles. I think it is due to the time to extract HOG features. I tried to reduce the time tuning the parameters of HOG features but it was not possible to process it realtime. 

I belive it would be better using deep learning than extracting HOG features.
