# iOS-YOLOv3tiny-Object-Detection-with-the-use-of-LiDAR-Depth-Estimation-Demonstration

Availability: This application only works on iPhone with LiDAR scanner hardware.

Environment tested: iPhone 12 Pro/ iOS 15.4.1/ Xcode version 13.2.1/ macOS Big Sur 11.6.5

This code was primarily a prototype of the Master's Thesis "Utilising the data from AI object detection and depth sensor to prevent visually impaired people from falling into railroad tracks" (preliminary title) at Graduate School of Information Technology, Kobe Institute of Computing.

(Drafting)

The ViewController file limits the frame per second into 3 units for saving memory and CPU resources of the iPhone 12 Pro. It accepts RGB images from the rear camera, then fixes the image resolution into 1920 × 1080 and reorient it into the right orientation so as to get the repeatable and correct results of object detection at all times.

Later, it commits object detection after sending the data to the detectorMain file. The detectorMain file contains a detector model pre-trained by the YOLOv3tiny algorithm which has the role of detecting the desired danger zone objects. It crops the images into 416 × 416 using scale fit function in order to fit the requirements of the YOLO algorithm.  The detector model accepts the cropped images and then predicts the desired objects in the images. 

The coordinates of the bounding boxes of the detected objects are used for drawing the bounding boxes of that on the iPhone screen (for debugging, not for the visually impaired users), and used for retrieving the LiDAR distance data. To make sure they are reflecting the same thing at the same time, the desired LiDAR depth images are retrieved from the same frame for object detection. In this prototype, the selected distance value lies on the closest grid of LiDAR depth image compared to the RGB’s one. By “closest”, it does not mean the coordinates are taken by other more precise calculations, rather, it is the distance value that corresponds to the point with minimum distance value of the object, i.e. bounding box. For the edge objects detected as a diagonal line [2], the application takes 10 points across the line to determine the point with minimum distance value, so as to define the closest area of an edge. Still, for the horizontal edges, the application would simply take the distance value of the midpoint to determine the closest area.

Finally, having both the information from object detection and distance detection, the application would give a verbal message output if there is a danger zone (in this case, platform edge) detected within 3m ahead of the user.

Ref:

[1]R, Okuda, AR Sample. <https://github.com/ryokuda/ARSample>.

[2]R, Okuda, "The pedestrian signal identification device," Japan Patent 6737968, Feb,07,2020. 
(奧田亮輔. 歩行者信号識別装置. 特許第6737968号. 2020-02-07.)

