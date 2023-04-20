from api import Detector
import os

# Initialize detector
detector = Detector(model_name='rapid',
                    weights_path='./weights/pL1_MWHB1024_Mar11_4000.ckpt',
                    use_cuda=True)

# A simple example to run on a single image and plt.imshow() it
detector.detect_one(img_path='./images/exhibition.jpg',
                    input_size=1024, conf_thres=0.3,
                    visualize=True)

# Experiment to loop it over multiple images
'''imageDir = './images'
for file in os.listdir(imageDir):
    if file.endswith(".jpg"):
        imagePath = os.path.join(imageDir, file)
        detector.detect_one(img_path=imagePath,
                            input_size=1024, conf_thres=0.3,
                            visualize=True)'''