from sklearn.model_selection import train_test_split
import os

def split_data(data_dir, test_size=0.2):
    # random split and just put the names in a txt file
    all_files = os.listdir(data_dir)
    train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42)
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)
    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for item in test_files:
            f.write("%s\n" % item)
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
    return train_files, test_files

if __name__ == "__main__":
    data_dir = "/home/waleed/Documents/3DLearning/margin-line/final_02/context_margin_colors_faces_classes"
    train_files, test_files = split_data(data_dir)
    print("Train files:", train_files)
    print("Test files:", test_files)