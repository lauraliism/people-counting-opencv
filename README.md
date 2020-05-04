# About
It ia a camera based people counter.   
This repo has been created based on [this blog post](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/). The difference is that instead of using the Single Shot Detector(SSD) for object detection this repository uses You Only Look Once ([YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)) deep learning algorithm.

It takes a video file (`.mp4`) and draws a counting line on it. Then it trys to first detect people in the frame, assigns an ID to them and starts tracking the objects.
 
# How to run it?

1. First install required libraries:  
  * [NumPy](https://numpy.org/)
  * [OpenCV](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/)
  * [dlib](http://dlib.net/)
  * [imutils](https://github.com/jrosebr1/imutils)
2. Then you need to acquire the pretrained weights file and put this file inside the `yolo-coco` folder:    
$ `wget https://pjreddie.com/media/files/yolov3.weights`
3. Run the detector:   
$ `python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 --output output/output_02.avi`
