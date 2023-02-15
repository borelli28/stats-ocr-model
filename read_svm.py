import cv2
import joblib


svm_model = joblib.load("./models/svm_model.pkl")

# Load the image
image_path = "./assets/labeled-images/jazz-chisholm-jr.png"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract the regions of the image containing text
text_regions = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    text_regions.append(image[y:y+h, x:x+w])

# Process each text region with your OCR model
for i, region in enumerate(text_regions):
    # Use your SVM OCR trained model to predict the labels for this region
    # labels = svm_model.predict(region)

    # Draw a rectangle around the text region
    contour = contours[i]
    if len(contour) >= 4:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add the predicted labels as text within the rectangle
        # label_text = ''.join(labels)
        # cv2.putText(image, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Display the image with text regions and labels
cv2.imshow("Text Regions with Labels", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
