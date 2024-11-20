import cv2 
import numpy as np  
import pandas as pd 

NUM_AT_A_TIME = 200
START_INDEX = 0

# Load and preprocess data
print("Loading image data...")
data = np.load('snipe_data_rgb_full.npz')['arr_0']
labels = data[:, 0].astype(np.int64)
metadata = data[:,1:8]
data = data[:,8:].astype(np.uint8)
data = data.reshape(-1, 256*2, 192*2, 3)

# load labels
label_names = pd.read_csv('labels.csv',header=None)
label_names = label_names.values.flatten()
label_names = label_names.flatten()

images = []
print(f"Processing {data.shape[0]} samples")
print("Select a ROI and space. Afterwards, 'd' for discard, 'n' for next, 'r' for redo, 'c' to exit the process.")
i = START_INDEX
while i < data.shape[0]:
    img = cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR)
    
    # Select ROI 
    r = cv2.selectROI(f"Select {label_names[labels[i]]}", img, printNotice=False) 
    
    # Crop image 
    cropped_image = img[int(r[1]):int(r[1]+r[3]),  
                        int(r[0]):int(r[0]+r[2])] 
    resized_image = cv2.resize(cropped_image, (256, 256))
    
    # Display cropped image 
    cv2.imshow("Cropped image", cropped_image) 

    k = cv2.waitKey(0)

    # discard
    if k == ord('d'):
        print("discard")

    # save and go onto next
    elif k == ord('n'):
        print("next")
        datum = [labels[i]]
        datum = np.concatenate((datum, metadata[i], resized_image.flatten()))
        images.append(datum)

    # redo this ROI
    elif k == ord('r'):
        print("redo")
        i -= 1
    
    # pressed wrong key
    else:
        print("Invalid keypress. 'd' for discard, 'n' for next, 'r' for redo, 'c' to exit the process.")
        i -= 1

    cv2.destroyAllWindows()
    i += 1

    if i % NUM_AT_A_TIME == 0:
        # parse data
        images = np.array(images)
        print(f"Saving {images.shape[0]} samples (indexes {i-NUM_AT_A_TIME} to {i})")
        unique, counts = np.unique(images[:,0], return_counts=True)
        counts_dict = dict(zip(label_names[unique], counts))
        print(counts_dict)

        # save data
        fname = f"snipe_data_cleaned_{int(i/NUM_AT_A_TIME)}.npz" 
        np.savez_compressed(fname, images)
        print(f"Saved to {fname}")

        images = []

images = np.array(images)
print(f"Ended with {images.shape[0]} samples")
unique, counts = np.unique(images[:,0], return_counts=True)
counts_dict = dict(zip(label_names[unique], counts))
print(counts_dict)

# save data
fname = f"snipe_data_cleaned_{int(i/NUM_AT_A_TIME) + 1}.npz"  
np.savez_compressed(fname, images)
print(f"Saved to {fname}")

