from datasets.image_dataset import OtitisMedia

dataset = OtitisMedia(csv_file= r'data/train.csv')
dataset = OtitisMedia(csv_file= r'data\train.csv')


print(f'this dataset has{len(dataset)} images/labels')
print(f'A sample from this dataset has shape {dataset[0][0].shape} and label {dataset[0][1]}')