import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Depth threshold values (in millimeters) - the depth we are interested in away from the camera
min_depth = 100  # minimum depth threshold in millimeters
max_depth = 1700  # maximum depth threshold in millimeters

# Define ROI dimensions - effectively crops the depth image
roi_x = 40  # X-coordinate of top-left corner of ROI
roi_y = 40  # Y-coordinate of top-left corner of ROI
roi_width = 500  # Width of ROI
roi_height = 350  # Height of ROI

# Define line of interest
line = [(250, 0), (250, 350)]

# Create OpenCV window
cv2.namedWindow("Person Detection", cv2.WINDOW_NORMAL)

# Create a dictionary to store object trackers using centroid tracking
object_trackers = {}

room_count = 0

while True:
    # Wait for a coherent pair of frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    colour_frame = frames.get_color_frame()
    if not depth_frame or not colour_frame:
        continue

    # Convert depth frame to numpy array
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert colour frame to numpy array
    colour_image = np.asanyarray(colour_frame.get_data())

    # Apply depth thresholding
    depth_mask = np.logical_and(
        depth_image > min_depth, depth_image < max_depth
    )
    depth_filtered = np.where(depth_mask, depth_image, 0)

    # Convert depth filtered image to 8-bit uint8 format
    depth_filtered_uint8 = cv2.convertScaleAbs(depth_filtered, alpha=0.3)

    # apply ROI to depth image to ignore image edges
    depth_filtered_uint8 = depth_filtered_uint8[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    colour_image = colour_image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

    # Find contours in the depth filtered image
    contours, hierarchy = cv2.findContours(
        depth_filtered_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Loop through contours
    for contour in contours:
        # Filter contours based on size, shape, etc.
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)

            # Estimate height of object using depth information
            person_depth = (np.median(depth_frame.get_distance(x + int(w / 2), y + h - 1))) * 1000

            # Filter person contours based on depth and size
            if person_depth > min_depth and person_depth < max_depth and h > 70:

                # Calculate centroid of the contour
                centroid_x = int(x + w / 2)
                centroid_y = int(y + h / 2)

                # Check if centroid is already being tracked, if not, add new tracker
                found = False
                for object_id, [tracker, (original_centroid_x, original_centroid_y)] in object_trackers.items():
                    success, box = tracker.update(depth_filtered_uint8)

                    tracked_x, tracked_y, tracked_w, tracked_h = box
                    tracked_centroid_x = int(tracked_x + tracked_w / 2)
                    tracked_centroid_y = int(tracked_y + tracked_h / 2)

                    # If the centroid of the contour is within the bounding box of a tracked object,
                    # update the tracker with the new centroid position and set found flag to True
                    if abs(tracked_centroid_x - centroid_x) < w / 2 and abs(tracked_centroid_y - centroid_y) < h / 2:
                        # Draw bounding box around person on depth image and label it
                        cv2.rectangle(depth_filtered_uint8, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(depth_filtered_uint8, str(object_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # check if the box moved past the line of interest:
                        # increment the room count
                        if original_centroid_x <= line[0][0] and tracked_centroid_x > line[0][0]:
                            room_count += 1

                        # decrement the room count
                        if original_centroid_x >= line[0][0] and tracked_centroid_x < line[0][0]:
                            room_count -= 1

                        # update the tracker for this contour with the new tracked bounding box
                        object_trackers[object_id][0].update(depth_filtered_uint8)
                        object_trackers[object_id][1] = (tracked_centroid_x, tracked_centroid_y)
                        found = True
                        break

                # If the contour does not match any existing trackers, create a new tracker for it
                if not found:
                    # Create a new tracker
                    tracker = cv2.TrackerCSRT_create()
                    
                    original_centroid_x = int(x + w / 2)
                    original_centroid_y = int(y + h / 2)

                    tracker.init(depth_filtered_uint8, (x, y, w, h))
                    # Add the new tracker to the dictionary with a unique object ID and the centroid of the bbox used to initialise it
                    object_id = len(object_trackers) + 1
                    object_trackers[object_id] = [tracker, (original_centroid_x, original_centroid_y)]
                    # Draw bounding box around person on depth image and label it
                    cv2.rectangle(depth_filtered_uint8, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(depth_filtered_uint8, str(object_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw the line of interest onto the frame
    cv2.line(depth_filtered_uint8, line[0], line[1], (255, 0, 0), 2)

    # Display room count
    cv2.putText(depth_filtered_uint8, "Room Count: " + str(room_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display depth filtered image with detected persons
    cv2.imshow("Person Detection", depth_filtered_uint8)
    cv2.imshow("Colour Image", colour_image)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()
pipeline.stop()
