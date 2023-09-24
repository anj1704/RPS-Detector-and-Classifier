import cv2
import os
import torch
import pickle
import numpy as np
from torchvision import transforms
import torch.nn as nn

class ObjectDetector(nn.Module):
	def __init__(self, baseModel, numClasses):
		super(ObjectDetector, self).__init__()
		# initialize the base model and the number of classes
		self.baseModel = baseModel
		self.numClasses = numClasses
		#self.identity = nn.Identity()
    # build the regressor head for outputting the bounding box
		# coordinates
		self.regressor = nn.Sequential(
			nn.Linear(1024, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 4),
			nn.Sigmoid()
		)
		self.classifier = nn.Sequential(
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, self.numClasses),
			nn.Softmax(dim=1)
		)
		self.baseModel.fc_model[-1] = nn.Identity()

	def forward(self, x):
      # pass the inputs through the base model and then obtain
      # predictions from two different branches of the network
			features = self.baseModel(x)
			#features = nn.Identity()(features)
			#features = nn.Identity()(self.baseModel(x))
			bboxes = self.regressor(features)
			classLogits = self.classifier(features)
			# return the outputs as a tuple
			return (bboxes, classLogits)
	
# Define the path to the base output directory
BASE_OUTPUT = "output"

# Define the paths for model, label encoder, plots output, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.490110532131934, 0.446009729818045, 0.43031791397747016]
STD = [0.21694581319104497, 0.20764065161771203, 0.20739287063608627]

# Specify the directory containing images
input_directory = "input_image"  # Change this to your directory path

# Load our object detector, set it to evaluation mode, and load the label encoder from disk
print("[INFO] Loading object detector...")
model = torch.load(MODEL_PATH, map_location=DEVICE).to(DEVICE)
model.eval()
le = pickle.loads(open(LE_PATH, "rb").read())

# Define normalization transforms
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Get a list of image file paths in the directory
imagePaths = [os.path.join(input_directory, filename) for filename in os.listdir(input_directory) if filename.endswith(".jpg")]

# Loop over the images in the directory and process each one
for imagePath in imagePaths:
    # Load the image, copy it, swap its color channels, resize it, and bring its channel dimension forward
    image = cv2.imread(imagePath)
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))

    # Convert image to PyTorch tensor, normalize it, move it to the current device, and add a batch dimension
    image = torch.from_numpy(image)
    image = transforms(image).to(DEVICE)
    image = image.unsqueeze(0)

    # Predict the bounding box of the object along with the class label
    (boxPreds, labelPreds) = model(image)

    # Debugging: Print predictions
    print("boxPreds:", boxPreds)
    print("labelPreds:", labelPreds)
	
    # Determine the class label with the largest predicted probability
    labelPreds = torch.nn.Softmax(dim=-1)(labelPreds)
    i = labelPreds.argmax(dim=-1).cpu()
    label = le.inverse_transform(i)[0]

    # Get the dimensions of the original image
    (h, w) = orig.shape[:2]

    # Scale the predicted bounding box coordinates based on the image dimensions
    startX, startY, width, height = boxPreds[0]
    startX = int(startX * w)
    startY = int(startY * h)
    width = int(width * w)
    height = int(height * h)

    # Calculate endX and endY based on startX, startY, width, and height
    endX = startX + width
    endY = startY + height

    # Draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Show the output image
    cv2.imshow("Output", orig)
	
    print(f"Image: {imagePath}")
    print(f"Predicted Class: {label}")
    print(f"Bounding Box Coordinates: (startX={startX}, startY={startY}, endX={endX}, endY={endY})")
    print("-" * 40)

    # Wait for a key press (5-second delay) and close the image window
    key = cv2.waitKey(5000) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

# Close all image windows
cv2.destroyAllWindows()
