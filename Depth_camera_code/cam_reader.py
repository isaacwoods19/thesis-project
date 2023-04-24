import pyrealsense2 as rs
import cv2
import numpy as np

# Create a RealSense context and set a devices changed callback
context = rs.context()
context.set_devices_changed_callback(lambda: pipeline.stop())

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# Set camera resolution and frame rate
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_device('846112071410')

# Start streaming
pipeline.start(config)

# Create a hole filling filter with 'nearest' mode
hole_filling_filter = rs.hole_filling_filter(2)

# Get the depth sensor
sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
# Find all possible visual presets
preset_range = sensor.get_option_range(rs.option.visual_preset)
# Select preset from list: Available presets: [Default, Hand, High Accuracy, High Density, Medium Density]
preset = 'High Density'

for i in range(int(preset_range.min), int(preset_range.max + 1)):
    description = sensor.get_option_value_description(rs.option.visual_preset, i)
    if description == preset:
        sensor.set_option(rs.option.visual_preset, i)
        break

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Apply hole filling to the depth frame
        depth_frame = hole_filling_filter.process(depth_frame)

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)
        
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
