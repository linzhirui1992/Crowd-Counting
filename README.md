# Crowd-Counting
Convolutional Network Demo For Crowd Counting


# Dependencies
python 3.6.2
opencv 3.4.5

# Run the demo
Example:
```
C:\Users\admin\Desktop\Crowd-Counting-master>python opencv_caffe_crowd_density_map.py
Input: (1, 3, 768, 1024) float32
inference image: 0.2005 seconds.
Output: (1, 1, 192, 256) float32
number:  285.41965
```

**input image:**  
![Alt text](https://github.com/linzhirui1992/Crowd-Counting/blob/master/IMG_191.jpg)  
**output result:**  
![Alt text](https://github.com/linzhirui1992/Crowd-Counting/blob/master/result.png)

# network structure
visualize the network via valid Caffe's prototext  
http://ethereon.github.io/netscope/#/editor

# reference
[A Deeply-Recursive Convolutional Network For Crowd Counting](https://arxiv.org/pdf/1805.05633.pdf)
