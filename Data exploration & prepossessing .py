# Extracting all filenames iteratively
base_path = 'COVID-19_Radiography_Dataset'
categories = ['COVID/images', 'Normal/images', 'Viral Pneumonia/images']


# load file names to fnames list object
fnames = []
for category in categories:
    image_folder = os.path.join(base_path, category)
    file_names = os.listdir(image_folder)
    full_path = [os.path.join(image_folder, file_name) for file_name in file_names]
    fnames.append(full_path)


print('number of images for each category:', [len(f) for f in fnames])
print(fnames[0:2]) #examples of file names

#Reduce number of images to first 1345 for each category
fnames[0] = fnames[0][0:1344]
fnames[1] = fnames[1][0:1344]
fnames[2] = fnames[2][0:1344]

# Import image, load to array of shape height, width, channels, then min/max transform.
# Write preprocessor that will match up with model's expected input shape.
from keras.preprocessing import image
import numpy as np
from PIL import Image



def preprocessor(img_path):
        img = Image.open(img_path).convert("RGB").resize((192,192)) # import image, make sure it's RGB and resize to height and width you want.
        img = (np.float32(img)-1.)/(255-1.) # min max transformation
        img=img.reshape((192,192,3)) # Create final shape as array with correct dimensions for Keras
        return img



#Try on single file (imports file and preprocesses it to data with following shape)
preprocessor('COVID-19_Radiography_Dataset/COVID/images/COVID-2273.png').shape

#Import image files iteratively and preprocess them into array of correctly structured data
# Create list of file paths
image_filepaths=fnames[0]+fnames[1]+fnames[2]


# Iteratively import and preprocess data using map function
# map functions apply your preprocessor function one step at a time to each filepath
preprocessed_image_data=list(map(preprocessor, image_filepaths))


# Object needs to be an array rather than a list for Keras (map returns to list object)
X = np.array(preprocessed_image_data) # Assigning to X to highlight that this represents feature input data for our model

print(len(X) )          # same number of elements as filenames

print(X.shape )         # dimensions now 192,192,3 for all images

print(X.min().round() ) # min value of every image is zero

print(X.max() )         # max value of every image is one

# Create y data made up of correctly ordered labels from file folders
from itertools import repeat


# Recall that we have three folders with the following number of images in each folder
#...corresponding to each type
print('number of images for each category:', [len(f) for f in fnames])
covid=list(repeat("COVID", 1344))
normal=list(repeat("NORMAL", 1344))
pneumonia=list(repeat("PNEUMONIA", 1344))


#combine into single list of y labels
y_labels = covid + normal + pneumonia


#check length, same as X above
print(len(y_labels))


# Need to one hot encode for Keras.  Let's use Pandas
import pandas as pd
y = pd.get_dummies(y_labels)


display(y)

print(type(X))
print('Number of elements: ', len(X) )
print('Dimensions for all images', X.shape )
print('Minimum value of every image: ', X.min().round() )
print('Maximum value of every image: ', X.max() )
print('Mean value of every image: ', X.mean() )
print('Standard deviation of every image: ', X.std() )

# Create a list of file paths for each category
category_filepaths = [fnames[0], fnames[1], fnames[2]]

# Initialize lists to store statistics for each category
category_statistics = []

# Iterate over each category
for filepaths in category_filepaths:
    # Preprocess images in the current category
    preprocessed_images = [preprocessor(filepath) for filepath in filepaths]

    # Convert list of preprocessed images to NumPy array
    preprocessed_images_array = np.array(preprocessed_images)

    # Compute statistics for the current category
    category_stats = {
        'mean': np.mean(preprocessed_images_array),
        'std': np.std(preprocessed_images_array),
        'min': np.min(preprocessed_images_array).round(),
        'max': np.max(preprocessed_images_array)
    }

    # Append statistics to the list
    category_statistics.append(category_stats)

# Print statistics for each category
for i, stats in enumerate(category_statistics):
    print(f"Statistics for Category {i+1}:")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Standard Deviation: {stats['std']:.4f}")
    print(f"Minimum Value: {stats['min']:.4f}")
    print(f"Maximum Value: {stats['max']:.4f}")
    print()

    # Define statistics names
stat_names = ['Mean', 'Standard Deviation', 'Minimum', 'Maximum']

# Define statistics values
mean_values = [stats['mean'] for stats in category_statistics]
std_values = [stats['std'] for stats in category_statistics]
min_values = [stats['min'] for stats in category_statistics]
max_values = [stats['max'] for stats in category_statistics]

# Define bar width
bar_width = 0.2
index = np.arange(len(categories))

# Plotting side-by-side bar graphs for each statistic
plt.figure(figsize=(10, 6))
plt.bar(index - bar_width * 1.5, mean_values, bar_width, label='Mean', color='blue')
plt.bar(index - bar_width * 0.5, std_values, bar_width, label='Standard Deviation', color='green')
plt.bar(index + bar_width * 0.5, min_values, bar_width, label='Minimum', color='orange')
plt.bar(index + bar_width * 1.5, max_values, bar_width, label='Maximum', color='red')

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Comparison of Statistics Across Categories')
plt.xticks(index, categories)
plt.legend()
plt.tight_layout()

# Show plot
plt.show()

from collections import Counter

# Count occurrences of each category
category_counts = Counter(y_labels)

# Extract categories and counts
cat = list(category_counts.keys())
cnt = list(category_counts.values())

# Plotting bar chart for each category
plt.figure(figsize=(8, 6))
plt.bar(cat, cnt, color=['blue', 'green', 'orange'])
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Counts per Category')
plt.show()

# Define the ImageDataGenerator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

aug_iter = datagen.flow(X, batch_size=1)

X_sample = X

# Apply augmentation directly to the original images stored in X_sample
num_augmented_images = 4032
aug_iter = datagen.flow(X_sample, batch_size=1)

# Overwrite original images in X with augmented images
for i in range(len(X_sample)):
    augmented_images = next(aug_iter)
    X_sample[i] = augmented_images[0]

# Verify the shape of X
print("Shape of X:", X_sample.shape)

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
# Visualize augmented images
plt.figure(figsize=(15, 10))
for i in range(9):
    augmented_images = next(aug_iter)
    augmented_image = array_to_img(augmented_images[0])
    plt.subplot(3, 3, i+1)
    plt.imshow(augmented_image)
    plt.title(f'Augmented Image {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()