from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os

# Set the path to the directory containing your original images
original_data_dir = r'data/tuber_dataset'

# Set the path to the directory where augmented images will be saved
augmented_data_dir = r'data/tuber_aug'

# Number of augmented images you want to generate per original image
augmentation_factor = 2  # Adjust this as needed

# Ensure the augmented data directory exists
os.makedirs(augmented_data_dir, exist_ok=True)

# Define data augmentation transformations
data_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ToTensor(),  # Convert the PIL Image to a Tensor
])

# Create a dataset using ImageFolder
dataset = ImageFolder(root=original_data_dir, transform=data_transform)

# Loop through each original image and generate augmented images
for i in range(len(dataset)):
    original_image, label = dataset[i]

    # Save the original image back to its class folder
    original_save_path = os.path.join(augmented_data_dir, dataset.classes[label], f'original_{i}.jpeg')
    os.makedirs(os.path.dirname(original_save_path), exist_ok=True)
    transforms.ToPILImage()(original_image).save(original_save_path)

    # Generate augmented images and save them to the corresponding class folder
    for j in range(augmentation_factor - 1):
        augmented_image, _ = dataset[i]  # Reapply the transformation to get a new augmented image
        augmented_save_path = os.path.join(augmented_data_dir, dataset.classes[label], f'augmented_{i}_{j}.jpeg')
        os.makedirs(os.path.dirname(augmented_save_path), exist_ok=True)
        transforms.ToPILImage()(augmented_image).save(augmented_save_path)

print("Augmentation complete.")
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

# Set the path to the directory containing your original images
original_data_dir = r'data/tuber_dataset'

# Set the path to the directory where augmented images will be saved
augmented_data_dir = r'data/tuber_aug'

# Number of augmented images you want to generate per original image
augmentation_factor = 2  # Adjust this as needed

# Ensure the augmented data directory exists
os.makedirs(augmented_data_dir, exist_ok=True)

# Define data augmentation transformations
data_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(256, scale=(0.9, 1.0)),
    transforms.ToTensor(),  # Convert the PIL Image to a Tensor
])

# Create a dataset using ImageFolder
dataset = ImageFolder(root=original_data_dir, transform=data_transform)

# Loop through each original image and generate augmented images
for i in range(len(dataset)):
    original_image, label = dataset[i]

    # Save the original image back to its class folder
    original_save_path = os.path.join(augmented_data_dir, dataset.classes[label], f'original_{i}.jpeg')
    os.makedirs(os.path.dirname(original_save_path), exist_ok=True)
    original_image.save(original_save_path)

    # Generate augmented images and save them to the corresponding class folder
    for j in range(augmentation_factor - 1):
        augmented_image, _ = dataset[i]  # Reapply the transformation to get a new augmented image
        augmented_save_path = os.path.join(augmented_data_dir, dataset.classes[label], f'augmented_{i}_{j}.jpeg')
        os.makedirs(os.path.dirname(augmented_save_path), exist_ok=True)
        augmented_image.save(augmented_save_path)

print("Augmentation complete.")
