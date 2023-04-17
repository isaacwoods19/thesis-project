import pyrealsense2 as rs
import cv2
import numpy as np

# Create a RealSense context and set a devices changed callback
context = rs.context()
context.set_devices_changed_callback(lambda: pipeline.stop())

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
pipeline.start(config)

# Distance threshold (in millimeters)
min_distance = 0  # Minimum distance in millimeters
max_distance = 1000  # Maximum distance in millimeters

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply the distance threshold to filter out values outside the range
        depth_image_filtered = np.where((depth_image >= min_distance) & (depth_image <= max_distance), depth_image, 0)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_filtered, alpha=0.2), cv2.COLORMAP_JET)
        
        # Convert color frame to OpenCV BGR format
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        # Display depth and color images
        cv2.imshow('Depth colour map', depth_colormap)
        cv2.imshow('Color Image', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
except KeyboardInterrupt:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
