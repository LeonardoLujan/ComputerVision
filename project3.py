import cv2
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

#config
CNN_IMG_SIZE = 64
MODEL_PATH = 'ball_classifier.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model definition
class BallClassifier(nn.Module):
    def __init__(self):
        super(BallClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def classify_ball_cnn(ball_patch, model, img_size=CNN_IMG_SIZE):
    #trying the CNN classification first
    try:
        #pre-process image for PyTorch
        img_resized = cv2.resize(ball_patch, (img_size, img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        #convert to tensor and normalize to [0, 1]
        tensor = torch.from_numpy(img_rgb).float() / 255.0
        
        #apply the ImageNet normalization (has to match the training)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - mean) / std
        
        tensor = tensor.unsqueeze(0).to(device)

        #make prediction
        model.eval()
        with torch.no_grad():
            prediction_tensor = model(tensor)
        
        score = prediction_tensor.item()
        
        #if model is confident, then just use it
        if score > 0.7 or score < 0.3:
            return 1 if score > 0.5 else 0
    except:
        pass
    
    #fallback to a traditional CV method
    return classify_ball_traditional(ball_patch)

def classify_ball_traditional(patch):
    #traditional CV classifier for when CNN fails or if it is uncertain
    if patch.size == 0:
        return 0
    
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    #create the circular mask
    h, w = patch.shape[:2]
    center = (w // 2, h // 2)
    radius = min(h, w) // 2 - 3
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    #edge detection to find stripes
    edges = cv2.Canny(gray, 40, 120)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    edge_density = np.sum(masked_edges > 0) / max(np.sum(mask > 0), 1)
    
    #texture variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_var = np.var(laplacian[mask > 0]) if np.sum(mask > 0) > 0 else 0
    
    #color variance
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    masked_hsv = hsv[mask > 0]
    if len(masked_hsv) > 0:
        hue_var = np.var(masked_hsv[:, 0])
    else:
        hue_var = 0
    
    #scoring: striped balls have more edges and texture variance
    score = 0
    if edge_density > 0.12:
        score += 2
    if texture_var > 400:
        score += 1
    if hue_var > 80:
        score += 1
    
    return 1 if score >= 2 else 0

def is_likely_cue_ball(patch):
    #check if patch is likely a cue ball (white ball to exclude)
    if patch.size == 0:
        return False
    
    #convert to HSV for better color analysis
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    
    #check brightness
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_patch)
    
    #check color saturation: cue balls have very low saturation
    mean_saturation = np.mean(hsv[:, :, 1])
    
    #cue ball criteria: bright with very low saturation
    #more lenient to catch cue balls in various lighting
    if mean_brightness > 200 and mean_saturation < 30:
        return True
    
    #also check for very uniform white regions
    std_brightness = np.std(gray_patch)
    if mean_brightness > 210 and std_brightness < 20:
        return True
    
    return False

def is_valid_ball_region(patch, center_x, center_y, radius, img_shape):
    #apply multiple filters to validate if detection is a real ball
    if patch.size == 0:
        return False
    
    #first: Check if the patch has a reasonable size
    if patch.shape[0] < 15 or patch.shape[1] < 15:
        return False
    
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    #second: perform variance check - balls should have texture
    std_dev = np.std(gray)
    if std_dev < 12:  #Slightly stricter
        return False
    
    #third: circularity check - balls should be round
    #Use contour detection to verify the shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            #circles have circularity close to 1.0
            if circularity < 0.5:  #too irregular
                return False
    
    #fourth: edge density check
    edges = cv2.Canny(gray, 30, 100)
    edge_ratio = np.sum(edges > 0) / edges.size
    
    #should have some edges (but not too many)
    if edge_ratio < 0.03 or edge_ratio > 0.5:
        return False
    
    #fifth: color/brightness validation
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_brightness = np.mean(gray)
    mean_saturation = np.mean(hsv[:, :, 1])
    
    #filter very dark regions (shadows/background)
    if mean_brightness < 40:
        return False
    
    #filter pure white or very bright low-saturation (likely table or glare)
    if mean_brightness > 230 and mean_saturation < 20:
        return False
    
    #sixth: Size consistency check
    h, w = patch.shape[:2]
    aspect_ratio = max(h, w) / max(min(h, w), 1)
    if aspect_ratio > 1.3:  #Balls should be roughly square patches
        return False
    
    return True

def process_image(image_path, model):
    #main function that detects and classifies billiard balls
    img = cv2.imread(image_path)
    if img is None:
        print("0")
        return

    #stage 1: detection using Hough Circle Transform
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #apply bilateral filter to reduce noise while preserving edges
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    #detect circles with balanced parameters
    circles = cv2.HoughCircles(
        gray_filtered,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=84,       #balance between the missing balls and the duplicates
        param1=50,
        param2=28,        #slightly stricter than 25
        minRadius=40,     #a more restrictive minimum
        maxRadius=130     #a more restrictive maximum
    )
    
    results = []
    
    if circles is not None:
        circles = np.around(circles[0, :]).astype(np.int32)
        
        for circle in circles:
            center_x, center_y, radius = circle[0], circle[1], circle[2]
            
            #basic radius validation
            if radius < 30 or radius > 150:
                continue
            
            #extract patch around detected circle
            buffer = 5
            y1 = max(0, center_y - radius - buffer)
            y2 = min(img.shape[0], center_y + radius + buffer)
            x1 = max(0, center_x - radius - buffer)
            x2 = min(img.shape[1], center_x + radius + buffer)

            #validate patch coordinates
            if x2 <= x1 or y2 <= y1:
                continue

            patch = img[y1:y2, x1:x2]

            if patch.size == 0:
                continue
            
            #apply validation filters
            if not is_valid_ball_region(patch, center_x, center_y, radius, img.shape):
                continue
            
            #filter out cue ball
            if is_likely_cue_ball(patch):
                continue
            
            #stage 2: Classification (solid vs striped)
            ball_type = classify_ball_cnn(patch, model)
            
            #append result: (X, Y, R, V)
            results.append((center_x, center_y, radius, ball_type))
    
    #output in the required format for submission
    print(len(results))
    for res in results:
        print(f"{res[0]} {res[1]} {res[2]} {res[3]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("0")
        sys.exit(1)
        
    image_file_path = sys.argv[1]

    #load trained PyTorch model
    try:
        classifier_model = BallClassifier().to(device)
        classifier_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        classifier_model.eval()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print("0")
        sys.exit(1)
        
    #run the actual detection
    process_image(image_file_path, classifier_model)
