#!/usr/bin/env python
# coding: utf-8

# ## Boxes with x axis to photo edges 

# In[53]:


import cv2
import numpy as np
import mediapipe as mp

"""
Boxes continue on a straight line right to the photo edges.

In the function process_landmarks_and_draw_bbox I have vert_factor, scale_factor for customizing 
and aligning the face mesh.
"""
# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_l.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Get the width of the image
image_width = frame.shape[1]

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

def bounding_box(landmarks):
    global min_y, max_y,max_y_region1,y_coords_max,y_coords_min,min_y_region2
    x_coords_min = [landmark[0] for landmark in landmarks]
    y_coords_min = [landmark[1] for landmark in landmarks]
    x_coords_max = [landmark[2] for landmark in landmarks]
    y_coords_max = [landmark[3] for landmark in landmarks]
    min_x = int(min(x_coords_min))
    min_y = int(min(y_coords_min))
    max_x = int(max(x_coords_max))
    max_y = int(max(y_coords_max))
    max_y_region1 = int(max([y for y in y_coords_max if y < max_y and y>= int(min(y_coords_max))]))
    min_y_region2 = int(max(y_coords_min))
    return [[min_x, min_y, max_x, max_y_region1], [min_x, min_y_region2, max_x, max_y]]

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame, vert_factor, scale_factor):
    global min_x, min_y, max_x, max_y
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor
    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    if min_x > 0:
        min_x = 0
    if max_x < image_width:
        max_x = image_width
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
    
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye",'left_eyebrow', "right_eye",'left_eyebrow'],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow":0}
scale_dict = {'lips': 1.2, 'left_eye': 2.2, 'left_eyebrow': 1.14, "right_eye": 2.2, "right_eyebrow":1.14}
# Process each set of landmarks and draw bounding box separately
box_coordinates = []
if results.multi_face_landmarks:
    for landmarks in results.multi_face_landmarks:
        bbox = process_landmarks_and_draw_bbox(frame, landmarks.landmark, frame.shape[1])
        if bbox:
            box_coordinates.append(bbox)

           
               
        # Extract bounding box and append the bounding box area to sections list
A = bounding_box(box_coordinates)
A[0][0],A[1][0] = 0,0
section_coords = [A[0], A[1]]

# Create the white spacer
spacer_height = section_coords[1][1] - section_coords[0][3]  # Correct height calculation
spacer_width = frame.shape[1]  # Use full width of the image
spacer = np.ones((spacer_height, spacer_width, 3), dtype=np.uint8) * 255  # White spacer

# Crop the sections between the white space
cropped_sections = []
for idx, coords in enumerate(section_coords):
    section = frame[coords[1]:coords[3], coords[0]:coords[2]]
    cropped_sections.append(section)
    print(idx)
    cv2.imwrite(fr"C:\Users\maorb\Desktop\Work\sad_2\cropped_{idx}.jpg", section)  # Saving each cropped section

# Concatenate sections and spacer to create the modified image
top_section = frame[section_coords[0][1]:section_coords[0][3], :]
bottom_section = frame[section_coords[1][1]:section_coords[1][3], :]
modified_image = np.vstack([top_section, spacer, bottom_section])

# Save the modified image with the white spacer
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\Now.jpg"  
cv2.imwrite(output_image_path, modified_image)



# Release resources
face_mesh.close()


# In[46]:


box_coordinates


# ## Boxes for all features

# In[52]:


import cv2
import numpy as np
import mediapipe as mp

"""
Boxes for each seperated feature

In the function process_landmarks_and_draw_bbox I have vert_factor, scale_factor for customizing 
and aligning the face mesh.
"""
# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\disgusted_4\photo_1.JPG"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame,vert_factor, scale_factor):
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor  # Adjust the scale factor to increase/reduce the stretching

    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
    
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye",'left_eyebrow', "right_eye",'left_eyebrow'],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow":0}
scale_dict = {'lips': 1.2, 'left_eye': 2.2, 'left_eyebrow': 1.14, "right_eye": 2.2, "right_eyebrow":1.14}
# Process each set of landmarks and draw bounding box separately
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])

# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\disgusted_4\Yes.JPG"

# Save the modified image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# In[38]:


landmarks


# ## Adaptive Boxes 

# In[8]:


import cv2
import numpy as np
import mediapipe as mp

"""
Adaptive boxes for each seperated feature

In the function process_landmarks_and_draw_bbox I have vert_factor, scale_factor for customizing 
and aligning the face mesh.
"""
# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\sad_li.jpeg"

# Read the input image
frame = cv2.imread(input_image_path)

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

def bounding_box(landmarks):
    global min_y, max_y,max_y_region1,y_coords_max,y_coords_min,min_y_region2
    x_coords_min = [landmark[0] for landmark in landmarks]
    y_coords_min = [landmark[1] for landmark in landmarks]
    x_coords_max = [landmark[2] for landmark in landmarks]
    y_coords_max = [landmark[3] for landmark in landmarks]
    min_x = int(min(x_coords_min))
    min_y = int(min(y_coords_min))
    max_x = int(max(x_coords_max))
    max_y = int(max(y_coords_max))
    max_y_region1 = int(max([y for y in y_coords_max if y < max_y and y>= int(min(y_coords_max))]))
    min_y_region2 = int(max(y_coords_min))
    return [[min_x, min_y, max_x, max_y_region1], [min_x, min_y_region2, max_x, max_y]]
# Function to process landmarks and draw bounding box

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame,vert_factor, scale_factor):
    global min_x,max_x, min_y, max_y
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor  # Adjust the scale factor to increase/reduce the stretching

    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)  # Blue color
    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362)
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)
]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye", "left_eyebrow", "right_eye", "right_eyebrow"],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow":0}
scale_dict = {'lips': 1.2, 'left_eye': 2, 'left_eyebrow': 1, "right_eye": 2, "right_eyebrow":1}
# Process each set of landmarks and draw bounding box separately
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])
# Process each set of landmarks and draw bounding box separately
box_coordinates = []
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame, vert_dict[contour_name], scale_dict[contour_name])
    box_coordinates.append((min_x, min_y, max_x, max_y))
    print(box_coordinates)
           
               
        # Extract bounding box and append the bounding box area to sections list
A = bounding_box(box_coordinates)
A[0][0],A[1][0] = 0,0
section_coords = [A[0], A[1]]

# Create the white spacer
spacer_height = section_coords[1][1] - section_coords[0][3]  # Correct height calculation
spacer_width = frame.shape[1]  # Use full width of the image
spacer = np.ones((spacer_height, spacer_width, 3), dtype=np.uint8) * 255  # White spacer

# Crop the sections between the white space
cropped_sections = []
for idx, coords in enumerate(section_coords):
    section = frame[coords[1]:coords[3], coords[0]:coords[2]]
    cropped_sections.append(section)
    cv2.imwrite(f"cropped_sectionss_{idx}.jpg", section)  # Saving each cropped section

# Concatenate sections and spacer to create the modified image
top_section = frame[section_coords[0][1]:section_coords[0][3], :]
bottom_section = frame[section_coords[1][1]:section_coords[1][3], :]
modified_image = np.vstack([top_section, spacer, bottom_section])

# Save the modified image with the white spacer
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\new_combinedYes.jpeg"  # Example name, replace with actual path
cv2.imwrite(output_image_path, modified_image)


# Output image path
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\maorAdaptive.jpeg"

# Save the modified image
cv2.imwrite(output_image_path, frame)

# Release resources
face_mesh.close()


# ## with upper cheek

# In[75]:


import cv2
import numpy as np
import mediapipe as mp

"""
Boxes continue on a straight line right to the photo edges.

In the function process_landmarks_and_draw_bbox I have vert_factor, scale_factor for customizing 
and aligning the face mesh.
"""
# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\scared_6\photo_1.JPG"

# Read the input image
frame = cv2.imread(input_image_path)

# Get the width of the image
image_width = frame.shape[1]

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

def bounding_box(landmarks):
    x_coords_min = [landmark[0] for landmark in landmarks]
    y_coords_min = [landmark[1] for landmark in landmarks]
    x_coords_max = [landmark[2] for landmark in landmarks]
    y_coords_max = [landmark[3] for landmark in landmarks]
    min_x = int(min(x_coords_min))
    min_y = int(min(y_coords_min))
    max_x = int(max(x_coords_max))
    max_y = int(max(y_coords_max))
    max_y_region1 = int(max([y for y in y_coords_max if y < max_y and y>= int(min(y_coords_max))]))
    min_y_region2 = int(max(y_coords_min))
    return [[min_x, min_y, max_x, max_y_region1], [min_x, min_y_region2, max_x, max_y]]

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame, vert_factor, scale_factor):
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor
    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    if min_x > 0:
        min_x = 0
    if max_x < image_width:
        max_x = image_width
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color
    return min_x, min_y, max_x, max_y
# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
    
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)] 
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]
right_upper_cheek= [(226,31),(31,228),(228,229),(229,230),(230,231),(231,232),(232,233),(233,244)]
left_upper_cheek = [(464,453),(453,452),(452,451),(451,450),(450,449),(449,448),(448,261),(261,446)]
# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye",'left_eyebrow', "right_eye",
                                              'right_eyebrow','left_upper_cheek','right_upper_cheek'],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, 
                                             right_eyebrow_contour,left_upper_cheek,right_upper_cheek]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, 
             "right_eyebrow":0,'left_upper_cheek': 0,'right_upper_cheek': 0}
scale_dict = {'lips': 1.2, 'left_eye': 2.2, 'left_eyebrow': 1.14, "right_eye": 1,
              "right_eyebrow":1.14, 'left_upper_cheek': 1,'right_upper_cheek': 1}
# Process each set of landmarks and draw bounding box separately
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])
box_coordinates = []
for contour_name, landmarks in landmarks_dict.items():
    A = process_landmarks_and_draw_bbox(landmarks, frame, vert_dict[contour_name], scale_dict[contour_name])
    box_coordinates.append((A))

           
               
        # Extract bounding box and append the bounding box area to sections list
A = bounding_box(box_coordinates)
A[0][0],A[1][0] = 0,0
section_coords = [A[0], A[1]]
output_image_path = r"C:\Users\maorb\Desktop\Work\scared_6\photo_1Boxes.JPG"

# Save the modified image
cv2.imwrite(output_image_path, frame)




# In[77]:


# Create the white spacer
input_image_path = r"C:\Users\maorb\Desktop\Work\scared_6\photo_1.JPG"

# Read the input image
frame = cv2.imread(input_image_path)
spacer_height = section_coords[1][1] - section_coords[0][3]  # Correct height calculation
spacer_width = frame.shape[1]  # Use full width of the image
spacer = np.ones((spacer_height, spacer_width, 3), dtype=np.uint8) * 255  # White spacer

# Crop the sections between the white space
cropped_sections = []
for idx, coords in enumerate(section_coords):
    section = frame[coords[1]:coords[3], coords[0]:coords[2]]
    cropped_sections.append(section)
    cv2.imwrite(fr"C:\Users\maorb\Desktop\Work\scared_6\\cropped_sections12_{idx}.jpg", section)  # Saving each cropped section

# Concatenate sections and spacer to create the modified image
top_section = frame[section_coords[0][1]:section_coords[0][3], :]
bottom_section = frame[section_coords[1][1]:section_coords[1][3], :]
modified_image = np.vstack([top_section, spacer, bottom_section])

# Save the modified image with the white spacer
output_image_path = r"C:\Users\maorb\Desktop\Work\scared_6\photo_1Cut.JPG"  # Example name, replace with actual path
cv2.imwrite(output_image_path, modified_image)




# # Final code

# In[68]:


import cv2
import numpy as np
import mediapipe as mp

"""
Boxes continue on a straight line right to the photo edges.

In the function process_landmarks_and_draw_bbox I have vert_factor, scale_factor for customizing 
and aligning the face mesh.
"""
# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Input image path
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\photo_12.JPG"

# Read the input image
frame = cv2.imread(input_image_path)

# Get the width of the image
image_width = frame.shape[1]

# Convert the image to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(frame_rgb)

def bounding_box(landmarks):
    global min_y, max_y,max_y_region1,y_coords_max,y_coords_min,min_y_region2
    x_coords_min = [landmark[0] for landmark in landmarks]
    y_coords_min = [landmark[1] for landmark in landmarks]
    x_coords_max = [landmark[2] for landmark in landmarks]
    y_coords_max = [landmark[3] for landmark in landmarks]
    min_x = int(min(x_coords_min))
    min_y = int(min(y_coords_min))
    max_x = int(max(x_coords_max))
    max_y = int(max(y_coords_max))
    max_y_region1 = int(max([y for y in y_coords_max if y < max_y and y>= int(min(y_coords_max))]))
    min_y_region2 = int(max(y_coords_min))
    return [[min_x, min_y, max_x, max_y_region1], [min_x, min_y_region2, max_x, max_y]]

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame, vert_factor, scale_factor):
    global min_x, min_y, max_x, max_y
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor
    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    if min_x > 0:
        min_x = 0
    if max_x < image_width:
        max_x = image_width
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # Optionally, draw the landmarks on the image
    for point in landmarks_scaled:
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color
    return min_x, min_y, max_x, max_y
# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
    
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]

# Extract landmarks coordinates for each feature
landmarks_dict = {}
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for i, lm in enumerate(face_landmarks.landmark):
            for contour_name, contour in zip(["lips", "left_eye",'left_eyebrow', "right_eye",'right_eyebrow'],
                                            [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour, right_eyebrow_contour]):
                if i in [idx for pair in contour for idx in pair]:
                    if contour_name not in landmarks_dict:
                        landmarks_dict[contour_name] = []
                    landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))


vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, "right_eyebrow":0}
scale_dict = {'lips': 1.2, 'left_eye': 2.2, 'left_eyebrow': 1.14, "right_eye": 2.2, "right_eyebrow":1.14}
# Process each set of landmarks and draw bounding box separately
for contour_name, landmarks in landmarks_dict.items():
    process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])
box_coordinates = []
for contour_name, landmarks in landmarks_dict.items():
    A = process_landmarks_and_draw_bbox(landmarks, frame, vert_dict[contour_name], scale_dict[contour_name])
    box_coordinates.append((A))

           
               
        # Extract bounding box and append the bounding box area to sections list
A = bounding_box(box_coordinates)
A[0][0],A[1][0] = 0,0
section_coords = [A[0], A[1]]
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\Po10.JPG"

# Save the modified image
cv2.imwrite(output_image_path, frame)




# In[69]:


# Create the white spacer
input_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\photo_12.JPG"

# Read the input image
frame = cv2.imread(input_image_path)
spacer_height = section_coords[1][1] - section_coords[0][3]  # Correct height calculation
spacer_width = frame.shape[1]  # Use full width of the image
spacer = np.ones((spacer_height, spacer_width, 3), dtype=np.uint8) * 255  # White spacer

# Crop the sections between the white space
cropped_sections = []
for idx, coords in enumerate(section_coords):
    section = frame[coords[1]:coords[3], coords[0]:coords[2]]
    cropped_sections.append(section)
    cv2.imwrite(fr"C:\Users\maorb\Desktop\Work\sad_2\cropped_sections12_{idx}.jpg", section)  # Saving each cropped section

# Concatenate sections and spacer to create the modified image
top_section = frame[section_coords[0][1]:section_coords[0][3], :]
bottom_section = frame[section_coords[1][1]:section_coords[1][3], :]
modified_image = np.vstack([top_section, spacer, bottom_section])

# Save the modified image with the white spacer
output_image_path = r"C:\Users\maorb\Desktop\Work\sad_2\Po20.jpeg"  # Example name, replace with actual path
cv2.imwrite(output_image_path, modified_image)




# In[ ]:


Revised version


# In[2]:


import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)


def bounding_box(landmarks):
    #global min_y, max_y,max_y_region1,y_coords_max,y_coords_min,min_y_region2
    x_coords_min = [landmark[0] for landmark in landmarks]
    y_coords_min = [landmark[1] for landmark in landmarks]
    x_coords_max = [landmark[2] for landmark in landmarks]
    y_coords_max = [landmark[3] for landmark in landmarks]
    min_x = int(min(x_coords_min))
    min_y = int(min(y_coords_min))
    max_x = int(max(x_coords_max))
    max_y = int(max(y_coords_max))
    max_y_region1 = int(max([y for y in y_coords_max if y < max_y and y>= int(min(y_coords_max))]))
    min_y_region2 = int(max(y_coords_min))
    return [[min_x, min_y, max_x, max_y_region1], [min_x, min_y_region2, max_x, max_y]]

# Function to process landmarks and draw bounding box
def process_landmarks_and_draw_bbox(landmarks, frame, vert_factor, scale_factor, image_width):
    #global min_x, min_y, max_x, max_y
    # Calculate centroid
    centroid_x = sum(point[0] for point in landmarks) / len(landmarks)
    centroid_y = sum(point[1] for point in landmarks) / len(landmarks)

    # Scale factor
    # Move the landmarks outward from the centroid
    landmarks_scaled = [
        ((point[0] - centroid_x) * scale_factor + centroid_x, 
         (point[1] - centroid_y) * scale_factor + centroid_y - vert_factor)
        for point in landmarks
    ]

    # Prepare for drawing
    pts = np.array(landmarks_scaled, np.int32)
    pts = pts.reshape((-1,1,2))

    # Draw a bounding box around the landmarks area
    min_x = int(min(point[0] for point in landmarks_scaled))
    max_x = int(max(point[0] for point in landmarks_scaled))
    min_y = int(min(point[1] for point in landmarks_scaled))
    max_y = int(max(point[1] for point in landmarks_scaled))
    if min_x > 0:
        min_x = 0
    if max_x < image_width:
        max_x = image_width
    # cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue color

    # # Optionally, draw the landmarks on the image
    # for point in landmarks_scaled:
    #     cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color
    
    return min_x, min_y, max_x, max_y

# List of landmark contours for each feature
lip_contours = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]
left_eye_contour = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), 
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), 
    (398, 362)
]
left_eyebrow_contour = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
right_eye_contour = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), 
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
right_eyebrow_contour = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]
right_upper_cheek= [(226,31),(31,228),(228,229),(229,230),(230,231),(231,232),(232,233),(233,244)]
left_upper_cheek = [(464,453),(453,452),(452,451),(451,450),(450,449),(449,448),(448,261),(261,446)]
def get_file_name_no_ext(file_path):
    # Get the base name of the file (i.e., name with extension)
    base_name = os.path.basename(file_path)
    # Split the base name into name and extension and return just the name
    return os.path.splitext(base_name)[0]


def crop_image(image_name, output_folder):
    # Read the input image
    frame = cv2.imread(image_name)

    # Get the width of the image
    image_width = frame.shape[1]

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect face landmarks
    results = face_mesh.process(frame_rgb)

    # Extract landmarks coordinates for each feature
    landmarks_dict = {}
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i, lm in enumerate(face_landmarks.landmark):
                for contour_name, contour in zip(["lips", "left_eye",'left_eyebrow', "right_eye",
                'right_eyebrow','left_upper_cheek','right_upper_cheek'],
                                                [lip_contours, left_eye_contour, left_eyebrow_contour, right_eye_contour,
                                                 right_eyebrow_contour,left_upper_cheek,right_upper_cheek]):
                    if i in [idx for pair in contour for idx in pair]:
                        if contour_name not in landmarks_dict:
                            landmarks_dict[contour_name] = []
                        landmarks_dict[contour_name].append((lm.x * frame.shape[1], lm.y * frame.shape[0]))

    vert_dict = {'lips': 5, 'left_eye': 5, 'left_eyebrow': 0, "right_eye": 5, 
    "right_eyebrow":0,'left_upper_cheek': 0,'right_upper_cheek': 0}
    scale_dict = {'lips': 1.2, 'left_eye': 2.2, 'left_eyebrow': 1.14, "right_eye": 1,
    "right_eyebrow":1.14, 'left_upper_cheek': 1,'right_upper_cheek': 1}
    # Process each set of landmarks and draw bounding box separately
    # for contour_name, landmarks in landmarks_dict.items():
    #     process_landmarks_and_draw_bbox(landmarks, frame,vert_dict[contour_name],scale_dict[contour_name])

    box_coordinates = []
    for contour_name, landmarks in landmarks_dict.items():
        min_x, min_y, max_x, max_y = process_landmarks_and_draw_bbox(landmarks, frame, vert_dict[contour_name], scale_dict[contour_name], image_width)
        box_coordinates.append((min_x, min_y, max_x, max_y))

    # Extract bounding box and append the bounding box area to sections list
    A = bounding_box(box_coordinates)
    A[0][0],A[1][0] = 0,0
    section_coords = [A[0], A[1]]

    base_name = get_file_name_no_ext(image_name)

    cropped_sections = []
    photo_type = ['eyes', 'lips']
    for idx, coords in enumerate(section_coords):
        section = frame[coords[1]:coords[3], coords[0]:coords[2]]
        cropped_sections.append(section)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_{photo_type[idx]}.PNG"), section)

crop_image(r"C:\Users\maorb\Desktop\Work\scared_6\photo_4.jpg",r"C:\Users\maorb\Desktop\Work\scared_6")



# In[9]:


output_directory = "C:/Users/maorb/Desktop/Experiment Builder/output"

def process_images_in_subfolders(output_directory):
    # Iterate over each folder in the output directory
    for root, dirs, files in os.walk(output_directory):
        for folder in dirs:
            # Extract the emotion abbreviation (e.g., "HA" for "Happiness")
            emotion_name = folder.split('-')[-1]
            emotion_abbr = emotion_name[:2].upper()

            # Get the folder path
            folder_path = os.path.join(root, folder)
            # Create a new folder for cropped images for this emotion if it doesn't exist
            cropped_folder_path = os.path.join(output_directory, folder + "_cropped")
            if not os.path.exists(cropped_folder_path):
                os.makedirs(cropped_folder_path)

            # Iterate over each file in the folder
            full_folder_path = os.path.join(root, folder)
            for file in os.listdir(full_folder_path):
                # Check if the file name ends with the emotion abbreviation
                if file.endswith(emotion_abbr + ".png"):  # Match file ending with "AB.jpg"
                    # Get the full path of the image file
                    image_path = os.path.join(full_folder_path, file)
                    # Call the crop_image function
                    #cropped_image_name = f"{os.path.splitext(file)[0]}_cropped.png"
                    #cropped_image_path = os.path.join(cropped_folder_path, cropped_image_name)
                    crop_image(image_path, cropped_folder_path)
# Call the function to process images in each subfolder
process_images_in_subfolders(output_directory)


# In[10]:


import os
import random
import csv

# Define the output directory path
output_directory = r'C:\Users\maorb\Desktop\Experiment Builder\output'

# Function to find folders ending with 'cropped' in the given directory
def find_cropped_folders(directory):
    cropped_folders = []
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            if folder.endswith('cropped'):
                cropped_folders.append(os.path.join(root, folder))
    return cropped_folders

# Function to get the number from the file name
def get_number_from_file_name(file_name):
    return int(file_name.split('-')[0][1:])

# Function to create random congruent and incongruent pairs
def create_pairs(cropped_folders, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Number', 'Emotion', 'Eyes', 'Lips', 'Congruence'])
        for folder in cropped_folders:
            emotion = os.path.basename(folder).split('_')[0]  # Extract emotion from folder name
            for file in os.listdir(folder):
                if file.endswith('eyes.PNG'):
                    number = get_number_from_file_name(file)
                    eyes_path = os.path.join(folder, file)
                    lips_path = os.path.join(folder, file.replace('eyes', 'lips'))
                    congruence = 1 if random.random() < 0.5 else 0
                    writer.writerow([number, emotion, eyes_path, lips_path, congruence])
                    # Choose random images from other emotion folders
                    for other_folder in cropped_folders:
                        if other_folder != folder:
                            other_emotion = os.path.basename(other_folder).split('_')[0]  # Extract emotion from folder name
                            other_files = [f for f in os.listdir(other_folder) if f.endswith(('eyes.PNG', 'lips.PNG'))]
                            random.shuffle(other_files)
                            chosen_file = None
                            for f in other_files:
                                if get_number_from_file_name(f) == number and (f != file and f != file.replace('eyes', 'lips')):
                                    chosen_file = os.path.join(other_folder, f)
                                    break
                            if chosen_file:
                                congruence = 1 if random.random() < 0.5 else 0
                                writer.writerow([number, other_emotion, chosen_file.replace('_eyes.PNG', '_eyes'), chosen_file.replace('_lips.PNG', '_lips'), congruence])

# Find all folders ending with 'cropped' in the output directory
cropped_folders = find_cropped_folders(output_directory)

# Specify the output CSV file path
output_csv = os.path.join(output_directory, 'congruence_data.csv')

# Create pairs and write to CSV
create_pairs(cropped_folders, output_csv)

print("CSV file created successfully!")


# In[12]:


import os
import random
import csv

# Define the path to the output directory
output_directory = r'C:\Users\maorb\Desktop\Experiment Builder\output'

# Initialize a list to store pairs of images for each number
image_pairs = []

# Iterate over each folder in the output directory
for root, dirs, files in os.walk(output_directory):
    for folder in dirs:
        # Check if the folder name ends with "_cropped"
        if folder.endswith("_cropped"):
            # Extract the number from the folder name
            number = folder.split('_')[0]

            # Initialize lists to store eyes and lips images for the number
            eyes_images = []
            lips_images = []

            # Get the path to the eyes and lips folders
            eyes_folder = os.path.join(root, folder)
            lips_folder = os.path.join(root, folder)

            # Iterate over files in the eyes folder
            for file in os.listdir(eyes_folder):
                if file.endswith("_eyes.PNG"):
                    eyes_images.append(os.path.join(eyes_folder, file))

            # Iterate over files in the lips folder
            for file in os.listdir(lips_folder):
                if file.endswith("_lips.PNG"):
                    lips_images.append(os.path.join(lips_folder, file))

            # Shuffle the lists to ensure random pairing
            random.shuffle(eyes_images)
            random.shuffle(lips_images)

            # Pair eyes and lips images for the number
            pairs_count = min(len(eyes_images), len(lips_images))
            for i in range(pairs_count):
                image_pairs.append((eyes_images[i], lips_images[i]))

# Shuffle the image pairs list
random.shuffle(image_pairs)

# Create and write the CSV file
csv_file_path = os.path.join(output_directory, "image_pairs.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Eyes Photo", "Lips Photo", "Congruence"])

    # Iterate over the image pairs
    for eyes_image, lips_image in image_pairs:
        # Determine congruence randomly (1 for congruent, 0 for incongruent)
        congruence = random.randint(0, 1)

        # Write the pair to the CSV file
        writer.writerow([eyes_image, lips_image, congruence])

print("CSV file created successfully:", csv_file_path)


# In[22]:


import os
import random
import csv

# Define the path to the output directory
output_directory = r'C:\Users\maorb\Desktop\Experiment Builder\output'

# Initialize a list to store pairs of images
image_pairs = []

# Iterate over each folder in the output directory
for root, dirs, files in os.walk(output_directory):
    for folder in dirs:
        # Check if the folder name ends with "_cropped"
        if folder.endswith("_cropped"):
            # Get the path to the eyes and lips folders
            eyes_folder = os.path.join(root, folder)
            lips_folder = os.path.join(root, folder)

            # Initialize lists to store eyes and lips images
            eyes_images = [img for img in os.listdir(eyes_folder) if img.endswith("_eyes.PNG")]
            lips_images = [img for img in os.listdir(lips_folder) if img.endswith("_lips.PNG")]

            # Shuffle the lists to ensure random pairing
            random.shuffle(eyes_images)
            random.shuffle(lips_images)

            # Pair eyes and lips images
            pairs_count = min(len(eyes_images), len(lips_images))
            for i in range(pairs_count):
                image_pairs.append((os.path.join(eyes_folder, eyes_images[i]), 
                                     os.path.join(lips_folder, lips_images[i])))

# Shuffle the image pairs list
random.shuffle(image_pairs)

# Create and write the CSV file
csv_file_path = os.path.join(output_directory, "image_pairs.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Eyes Photo", "Lips Photo", "Congruence"])

    # Iterate over the image pairs
    for eyes_image, lips_image in image_pairs:
        # Get the numbers from the filenames
        eyes_number = os.path.basename(eyes_image).split('-')[0][0:3]
        lips_number = os.path.basename(lips_image).split('-')[0][0:3]

        # Determine congruence based on the numbers
        congruence = 1 if eyes_number == lips_number else 0

        # Write the pair to the CSV file
        writer.writerow([eyes_image, lips_image, congruence])

print("CSV file created successfully:", csv_file_path)


# In[26]:


import os
import random
import csv

# Define the path to the output directory
output_directory = r'C:\Users\maorb\Desktop\Experiment Builder\output'

# Initialize a dictionary to store pairs of images for each number
image_pairs_by_number = {}

# Iterate over each folder in the output directory
for root, dirs, files in os.walk(output_directory):
    for folder in dirs:
        # Check if the folder name ends with "_cropped"
        if folder.endswith("_cropped"):
            # Get the number from the folder name

            # Initialize lists to store eyes and lips images for the number
            eyes_images = []
            lips_images = []

            # Get the path to the eyes and lips folders
            eyes_folder = os.path.join(root, folder)
            lips_folder = os.path.join(root, folder)

            # Iterate over files in the eyes folder
            for file in os.listdir(eyes_folder):
                if file.endswith("_eyes.PNG"):
                    eyes_images.append(os.path.join(eyes_folder, file))

            # Iterate over files in the lips folder
            for file in os.listdir(lips_folder):
                if file.endswith("_lips.PNG"):
                    lips_images.append(os.path.join(lips_folder, file))

            # Shuffle the lists to ensure random pairing
            random.shuffle(eyes_images)
            random.shuffle(lips_images)
            eyes_number = os.path.basename(eyes_image).split('-')[0][0:3]
            lips_number = os.path.basename(lips_image).split('-')[0][0:3]
            numbers = range(36)
            num_list = [eyes_number in numbers if eyes_number == lips_number]
            # Pair eyes and lips images for the number
            #pairs_count = min(len(eyes_images), len(lips_images))
            for i in range(pairs_count):
                if number not in image_pairs_by_number:
                    image_pairs_by_number[number] = []
                image_pairs_by_number[number].append((eyes_images[i], lips_images[i]))




# In[29]:


import os
import random
import csv

# Define the path to the output directory
output_directory = r'C:\Users\maorb\Desktop\Experiment Builder\output'

# Initialize a list to store the final image pairs
final_image_pairs = []

# Iterate over each folder in the output directory
for root, dirs, files in os.walk(output_directory):
    for folder in dirs:
        # Check if the folder name ends with "_cropped"
        if folder.endswith("_cropped"):
            # Get the emotion from the folder name
            emotion = folder.split('-')[1]

            # Initialize lists to store eyes and lips images
            eyes_images = []
            lips_images = []

            # Get the path to the eyes and lips folders
            eyes_folder = os.path.join(root, folder)
            lips_folder = os.path.join(root, folder)

            # Iterate over files in the eyes folder
            for file in os.listdir(eyes_folder):
                if file.endswith("_eyes.PNG"):
                    eyes_images.append(os.path.join(eyes_folder, file))

            # Iterate over files in the lips folder
            for file in os.listdir(lips_folder):
                if file.endswith("_lips.PNG"):
                    lips_images.append(os.path.join(lips_folder, file))

            # Shuffle the lists to ensure random pairing
            random.shuffle(eyes_images)
            random.shuffle(lips_images)

            # Pair eyes and lips images for the number
            pairs_count = min(len(eyes_images), len(lips_images))
            for i in range(pairs_count):
                # Sample congruency (1 for congruent, 0 for incongruent)
                congruency = random.choice([0, 1])

                # For congruent pairs, select random image pair from different emotions
                if congruency == 1:
                    random_emotion = random.choice([e for e in dirs if e != folder])
                    random_eyes_image = random.choice(os.listdir(os.path.join(root, random_emotion)))
                    random_lips_image = random.choice(os.listdir(os.path.join(root, random_emotion)))
                    final_image_pairs.append((os.path.join(eyes_folder, eyes_images[i]),
                                              os.path.join(lips_folder, lips_images[i]),
                                              congruency))
                    final_image_pairs.append((os.path.join(root, random_emotion, random_eyes_image),
                                              os.path.join(root, random_emotion, random_lips_image),
                                              congruency))
                # For incongruent pairs, select random image pair from different folders
                else:
                    random_eyes_folder = random.choice([f for f in dirs if f != folder])
                    random_lips_folder = random.choice([f for f in dirs if f != folder])
                    random_eyes_image = random.choice(os.listdir(os.path.join(root, random_eyes_folder)))
                    random_lips_image = random.choice(os.listdir(os.path.join(root, random_lips_folder)))
                    final_image_pairs.append((os.path.join(eyes_folder, eyes_images[i]),
                                              os.path.join(lips_folder, lips_images[i]),
                                              congruency))
                    final_image_pairs.append((os.path.join(root, random_eyes_folder, random_eyes_image),
                                              os.path.join(root, random_lips_folder, random_lips_image),
                                              congruency))

# Shuffle the final image pairs list
random.shuffle(final_image_pairs)

# Create and write the CSV file
csv_file_path = os.path.join(output_directory, "image_pairs.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Eyes Path", "Lips Path", "Congruency"])

    # Iterate over the final image pairs
    for eyes_image, lips_image, congruency in final_image_pairs:
        # Write the pair to the CSV file
        writer.writerow([eyes_image, lips_image, congruency])

print("CSV file created successfully:", csv_file_path)


# In[51]:


import os
import random
import csv

# Define the path to the output directory
output_directory = r'C:\Users\maorb\Desktop\Experiment Builder\output_cropped'

# Initialize a list to store the final image pairs
final_image_pairs = []

# Iterate over each folder in the output directory
for root, dirs, files in os.walk(output_directory):
    for folder in dirs:
        # Check if the folder name ends with "_cropped"
        if folder.endswith("_cropped"):
            # Get the emotion from the folder name
            emotion = folder.split('-')[1]
                       # Initialize lists to store eyes and lips images
            eyes_images = []
            lips_images = []
            # Get the path to the eyes and lips folders
            eyes_folder = os.path.join(root, folder)
            lips_folder = os.path.join(root, folder)
            for file in os.listdir(eyes_folder):
                if file.endswith("_eyes.PNG"):
                    eyes_images.append(os.path.join(eyes_folder, file))

            # Iterate over files in the lips folder
            for file in os.listdir(lips_folder):
                if file.endswith("_lips.PNG"):
                    lips_images.append(os.path.join(lips_folder, file))
            # Shuffle the lists to ensure random pairing
            random.shuffle(eyes_images)
            random.shuffle(lips_images)
            cong_list = [1] * 232 + [0] * 232
            random.shuffle(cong_list)
            # Pair eyes and lips images for the number
            for i in range(len(cong_list)):
                # Sample congruency (1 for congruent, 0 for incongruent)
                # For congruent pairs, select random image pair from different emotions
                if cong_list[i] == 1:
                    updated_eye_images = [m for m in eyes_images if m not in final_image_pairs[m][0]]
                    updated_lips_images = [m for m in lips_images if m not in final_image_pairs[m][1]]
                    eyes_image = random.choice(updated_eye_images)
                    num_match = eyes_image.split('-')[0][0:3]
                    emotion = eyes_image.split('\\')[6].split('-')[1]
                    filtered_lips_images = [a for a in updated_lips_images if a.split('-')[0][0:3] == num_match and
                                            a.split('\\')[6].split('-')[1] != emotion]
                    lips_image = random.choice(filtered_lips_images)
                    final_image_pairs.append(eyes_image,lips_image, cong_list[i])
                else:
                    updated_eye_images = [m for m in eyes_images if m not in final_image_pairs]
                    updated_lips_images = [m for m in lips_images if m not in final_image_pairs]
                    eyes_image = random.choice(updated_eye_images)
                    num_match = eyes_image.split('-')[0][0:3]
                    filtered_lips_images = [a for a in updated_lips_images if a.split('-')[0][0:3] != num_match]


# # Last code

# In[192]:


import os
import random
import csv

# Define the path to the output directory
output_directory = r'C:\Users\maorb\Desktop\Experiment Builder\output_cropped'

# Initialize a list to store the final image pairs
final_image_pairs = []
eyes_images = []
lips_images = []
# Iterate over each folder in the output directory
for root, dirs, files in os.walk(output_directory):
    for folder in dirs:
        # Check if the folder name ends with "_cropped"
        if folder.endswith("_cropped"):
            # Get the emotion from the folder name
            emotion = folder.split('-')[1]

            # Initialize lists to store eyes and lips images
            

            # Get the path to the eyes and lips folders
            eyes_folder = os.path.join(root, folder)
            lips_folder = os.path.join(root, folder)

            # Iterate over files in the eyes folder
            for file in os.listdir(eyes_folder):
                if file.endswith("_eyes.PNG"):
                    eyes_images.append(os.path.join(eyes_folder, file))

            # Iterate over files in the lips folder
            for file in os.listdir(lips_folder):
                if file.endswith("_lips.PNG"):
                    lips_images.append(os.path.join(lips_folder, file))

            # Shuffle the lists to ensure random pairing
            random.shuffle(eyes_images)
            random.shuffle(lips_images)
cong_list = [1] * 129 + [0] * 129
random.shuffle(cong_list)
#for i in range(min(len(eyes_images), len(lips_images))):
for i in cong_list:
    # Sample congruency (1 for congruent, 0 for incongruent)
    #congruence = random.choice([0, 1])
    # For congruent pairs, select random image pair from different emotions
    if i == 1:
        updated_eye_images = [m for m in eyes_images if all(m not in pair[0] for pair in final_image_pairs)]
        updated_lip_images = [m for m in lips_images if all(m not in pair[1] for pair in final_image_pairs)]
        eyes_image = random.choice(updated_eye_images)
        num_match = os.path.basename(eyes_image).split('-')[0][0:3]
        emotion = os.path.basename(eyes_image).split('-')[2][0]
        filtered_lips_images = [lip for lip in updated_lip_images if os.path.basename(lip).split('-')[0][0:3] == num_match
                                and os.path.basename(lip).split('-')[2][0] != emotion]

        if filtered_lips_images:
            lips_image = random.choice(filtered_lips_images)
            final_image_pairs.append((eyes_image, lips_image, i))
            
    else:
        updated_eye_images = [m for m in eyes_images if all(m not in pair[0] for pair in final_image_pairs)]
        updated_lip_images = [m for m in lips_images if all(m not in pair[1] for pair in final_image_pairs)]
        eyes_image = random.choice(updated_eye_images)
        num_match1 = os.path.basename(eyes_image).split('-')[0][0:3]
        filtered_lips_images = [lip for lip in updated_lip_images if os.path.basename(lip).split('-')[0][0:3] != num_match1]
        if filtered_lips_images:
            lips_image = random.choice(filtered_lips_images)
            final_image_pairs.append((eyes_image, lips_image, i))


# Create and write the CSV file
csv_file_path = os.path.join(output_directory, "image_pairs1.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Eyes_Path", "Lips_Path", "Congruency"])
    for pair in final_image_pairs:
        writer.writerow(pair)

print("CSV file created successfully:", csv_file_path)


# # sanity checks

# In[182]:


df = pd.DataFrame(final_image_pairs, columns=["Eyes_Path", "Lips_Path", "Congruency"])

# Splitting the values in the "Eyes_Path" column
df['Eyes_Num_Match'] = df['Eyes_Path'].apply(lambda x: os.path.basename(x).split('-')[0][0:3])

# Splitting the values in the "Lips_Path" column
df['Lips_Num_Match'] = df['Lips_Path'].apply(lambda x: os.path.basename(x).split('-')[0][0:3])

# Splitting the values in the "Eyes_Path" column to get the emotion
df['Eyes_Emotion'] = df['Eyes_Path'].apply(lambda x: os.path.basename(x).split('-')[2][0])

# Splitting the values in the "Lips_Path" column to get the emotion
df['Lips_Emotion'] = df['Lips_Path'].apply(lambda x: os.path.basename(x).split('-')[2][0])

A = df[(df['Congruency'] == 0) & (df['Eyes_Num_Match'] == df['Lips_Num_Match'])]
A


# # Correct CSV

# In[201]:


import os
import random
import csv

# Define the path to the output directory
output_directory = r'C:\Users\maorb\Desktop\Experiment Builder\output_cropped'

# Initialize a list to store the final image pairs
final_image_pairs = []
eyes_images = []
lips_images = []
# Iterate over each folder in the output directory
for root, dirs, files in os.walk(output_directory):
    for folder in dirs:
        # Check if the folder name ends with "_cropped"
        if folder.endswith("_cropped"):
            # Get the emotion from the folder name
            emotion = folder.split('-')[1]

            # Initialize lists to store eyes and lips images
            

            # Get the path to the eyes and lips folders
            eyes_folder = os.path.join(root, folder)
            lips_folder = os.path.join(root, folder)

            # Iterate over files in the eyes folder
            for file in os.listdir(eyes_folder):
                if file.endswith("_eyes.PNG"):
                    eyes_images.append(os.path.join(eyes_folder, file))

            # Iterate over files in the lips folder
            for file in os.listdir(lips_folder):
                if file.endswith("_lips.PNG"):
                    lips_images.append(os.path.join(lips_folder, file))

            # Shuffle the lists to ensure random pairing
            random.shuffle(eyes_images)
            random.shuffle(lips_images)
cong_list = [1] * 151 + [0] * 151
random.shuffle(cong_list)
for i in cong_list:

    if i == 1:
        updated_eye_images = [m for m in eyes_images if all(m not in pair[1] for pair in final_image_pairs)]
        updated_lip_images = [m for m in lips_images if all(m not in pair[2] for pair in final_image_pairs)]
        eyes_image = random.choice(updated_eye_images)
        num_match = os.path.basename(eyes_image).split('-')[0][0:3]
        emotion = os.path.basename(eyes_image).split('-')[2][0]
        filtered_lips_images = [lip for lip in updated_lip_images if os.path.basename(lip).split('-')[0][0:3] == num_match
                                and os.path.basename(lip).split('-')[2][0] == emotion]

        if filtered_lips_images:
            lips_image = random.choice(filtered_lips_images)
            final_image_pairs.append((num_match,eyes_image, lips_image, i))
            
    else:
        updated_eye_images = [m for m in eyes_images if all(m not in pair[1] for pair in final_image_pairs)]
        updated_lip_images = [m for m in lips_images if all(m not in pair[2] for pair in final_image_pairs)]
        eyes_image = random.choice(updated_eye_images)
        num_match = os.path.basename(eyes_image).split('-')[0][0:3]
        emotion = os.path.basename(eyes_image).split('-')[2][0]
        filtered_lips_images = [lip for lip in updated_lip_images if os.path.basename(lip).split('-')[0][0:3] == num_match
                                and os.path.basename(lip).split('-')[2][0] != emotion]        
        if filtered_lips_images:
            lips_image = random.choice(filtered_lips_images)
            final_image_pairs.append((num_match,eyes_image, lips_image, i))


# Create and write the CSV file
csv_file_path = os.path.join(output_directory, "image_last.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id',"Eyes_Path", "Lips_Path", "Congruency"])
    for pair in final_image_pairs:
        writer.writerow(pair)

print("CSV file created successfully:", csv_file_path)

