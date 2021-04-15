# Genetic Algorithm

This implementation of Genetic Algorithm recreates the input image using circles as chromosomes. It resizes the image for optimization

___________
# Examples

|        |        |        |        |
|--------|--------|--------|--------|
| ![wave_result](/examples/wave_result.png) | ![pixel_result](/examples/pixel_result.png) | ![catty_result](/examples/catty_result.png) | ![sunset_result](/examples/sunset_result.png) |
__________________________
## 1) Pre-running the code

The algorithm is written Python language using several external libraries.
Before running the test function please make sure, that *cv2* and *numpy* are installed, otherwise use pip in terminal or cmd to install them.

    pip install opencv-python
  
    pip install numpy
________________________________________________
## 2) Instructions on how to use the test function:

* Upload the input image to the same folder as the algorithm. 
(File format should preferably be .jpg or .png)

* Run the code

* Input the file name in user input with the extention without any additional characters:

```Input: pic.jpg```


* If the input image is not in the same folder, then input the entire file path to the input image


```Input: C:\Users\Mango\Pictures\pic.jpg```

* For the duration of the execution the code will print which generation it has currently done. Every 100th generation it will save an image to the same folder for you to see the progress. The name of those images will display the number of generation and the fitness of the image.
_________________________________________________

