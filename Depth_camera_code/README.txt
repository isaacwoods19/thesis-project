This section holds the code for my experiments around the Intel Realsense D435i depth camera and counting people walking through a threshold.
This code and environment had to be created from within a Linux environment due to the improved support and quality for the Intel Realsense SDK compared to the Windows version.

This code provides a solution to counting and tracking people in an enclosed space without the use of any artificial intelligence. Instead, it makes use of a depth camera and various algorithms.
This detection is done by settings a depth that the camera observes, ignoring distant data like the floor, to isolate the head and shoulders of people.
These isolated 'blobs' are referred to as 'contours', and are identified as people depending on size and distance from the camera.

These contours are tracked across the frame by finding the center point of a new bounding box and seeing if it falls in the area of a previous bounding box,
If it does, then this bounding box is tracked and treated like the first box has just moved.

It then simply gets the new and old bounding boxes and checks to see if they have passed over a set line in the frame, and manipulates the room count accordingly.

### TODO ###

Look at how I can improve the detection quality and reduce drop outs