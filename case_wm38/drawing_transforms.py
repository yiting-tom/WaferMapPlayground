# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from tqdm import tqdm

# %%
npz = np.load("./Wafer_Map_Datasets.npz", allow_pickle=True)
images: np.ndarray = npz["arr_0"]
labels: np.ndarray = npz["arr_1"]


# %%
def sparse_image_preserve_positions(
    original_image: np.ndarray, target_size: tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    sparse image from original size to target size, keeping the relative positions of points with value 2

    Parameters:
    original_image: numpy array, original image data (52x52)
    target_size: tuple, target size (256, 256)

    Returns:
    sparsed_image: numpy array, sparsed image
    """
    original_height, original_width = original_image.shape
    target_height, target_width = target_size

    # calculate scaling factors
    scale_y = target_height / original_height
    scale_x = target_width / original_width

    # create new empty image
    sparsed_image = np.zeros(target_size)

    # find all points with value 2
    yellow_points = np.where(original_image == 2)

    # map each yellow point to new positions
    for i in range(len(yellow_points[0])):
        old_y, old_x = yellow_points[0][i], yellow_points[1][i]

        # calculate new positions (keep relative positions)
        new_y = int(old_y * scale_y)
        new_x = int(old_x * scale_x)

        # ensure not out of bounds
        new_y = min(new_y, target_height - 1)
        new_x = min(new_x, target_width - 1)

        # set yellow points at new positions
        sparsed_image[new_y, new_x] = 2

    # for other values (0 and 1), you can choose different strategies
    # here we use nearest neighbor interpolation to fill background

    # create a mask to mark the positions where yellow points have been placed
    yellow_mask = sparsed_image == 2

    # resize original image to target size
    background = cv2.resize(
        original_image.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(float)

    # set yellow points in background to 0 (avoid overlap)
    background[background == 2] = 0

    # combine background and yellow points
    sparsed_image = np.where(yellow_mask, sparsed_image, background)

    return sparsed_image


def upscale_corners(image, target_value=1):
    image = image.astype(np.uint8)
    upscaled = np.zeros_like(image, dtype=np.uint8)

    # Check 4 corner patterns for where original image == 0
    cond = image == 0

    # left-top: (i,j+1)==1 and (i+1,j)==1
    cond_left_top = np.zeros_like(image, dtype=bool)
    cond_left_top[:-1, :-1] = (image[:-1, 1:] == target_value) & (
        image[1:, :-1] == target_value
    )

    # right-top: (i,j-1)==1 and (i+1,j-1)==1
    cond_right_top = np.zeros_like(image, dtype=bool)
    cond_right_top[:-1, 1:] = (image[:-1, :-1] == target_value) & (
        image[1:, :-1] == target_value
    )

    # left-bottom: (i-1,j)==1 and (i-1,j+1)==1
    cond_left_bottom = np.zeros_like(image, dtype=bool)
    cond_left_bottom[1:, :-1] = (image[:-1, :-1] == target_value) & (
        image[:-1, 1:] == target_value
    )

    # right-bottom: (i-1,j)==1 and (i-1,j-1)==1
    cond_right_bottom = np.zeros_like(image, dtype=bool)
    cond_right_bottom[1:, 1:] = (image[:-1, 1:] == target_value) & (
        image[:-1, :-1] == target_value
    )

    corner_fill = cond & (
        cond_left_top | cond_right_top | cond_left_bottom | cond_right_bottom
    )

    upscaled[corner_fill] = 1
    upscaled[image == target_value] = target_value

    return upscaled


# %%
# shape: (200, 200)
sparsed = sparse_image_preserve_positions(images[0], (200, 200))

# shape: (256, 256)
zoom_out = cv2.copyMakeBorder(
    sparsed, 28, 28, 28, 28, cv2.BORDER_CONSTANT, value=(0, 0, 0)
)
# shape: (256, 256)
zoom_out[zoom_out == 1] = 0
structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
zoom_out = ndimage.binary_dilation(zoom_out, structure=structure).astype(np.uint8)
plt.imshow(sparsed)

# %%
upscaled = upscale_corners(zoom_out)
plt.imshow(upscaled)
plt.colorbar()
# %%
circle = np.zeros((256, 256)).astype(np.uint8)
cv2.circle(circle, (127, 127), 100, (1, 1, 1), -1)

upscaled
filtered = cv2.filter2D(upscaled, -1, np.ones((2, 2)))
filtered = (filtered > 0).astype(np.uint8)
plt.imshow(filtered)
plt.colorbar()
# %%
with_circle = circle + filtered
# %%
zoom_in = with_circle[26:230, 26:230]
size_52 = cv2.resize(zoom_in.copy(), (52, 52), interpolation=cv2.INTER_NEAREST)
plt.imshow(size_52)
plt.colorbar()

# %%
# plot original, sparsed, extended, upscaled, filtered
plt.figure(figsize=(20, 10))

plt.subplot(2, 4, 1)
plt.imshow(images[0])
plt.title("Original")

plt.subplot(2, 4, 2)
plt.imshow(sparsed)
plt.title("Sparsed")

plt.subplot(2, 4, 3)
plt.imshow(zoom_out)
plt.title("Zoom Out")

plt.subplot(2, 4, 4)
plt.imshow(upscaled)
plt.title("Upscaled")

plt.subplot(2, 4, 5)
plt.imshow(filtered)
plt.title("Filtered")

plt.subplot(2, 4, 6)
plt.imshow(with_circle)
plt.title("With Circle")

plt.subplot(2, 4, 7)
plt.imshow(zoom_in)
plt.title("Zoom In")

plt.subplot(2, 4, 8)
plt.imshow(size_52)
plt.title("Size 52")

# %%


def upscaled_filtered_with_circle_zoom_in(image: np.ndarray) -> np.ndarray:
    image[image == 1] = 0
    image[image == 2] = 1
    upscaled = upscale_corners(image, target_value=1)
    filtered = cv2.filter2D(upscaled, -1, np.ones((5, 5)))
    filtered = (filtered > 0).astype(np.uint8)
    circle = np.zeros((256, 256)).astype(np.uint8)
    circle = cv2.circle(circle, (127, 127), 100, (1, 1, 1), -1)
    with_circle = circle + filtered
    zoom_in = with_circle[26:230, 26:230]
    zoom_in = cv2.resize(zoom_in.copy(), (52, 52), interpolation=cv2.INTER_NEAREST)
    return zoom_in.astype(np.uint8)


def wm38_to_sparse_draw(image: np.ndarray) -> np.ndarray:
    target_size = (200, 200)
    sparsed = sparse_image_preserve_positions(image, target_size)
    sparsed[sparsed == 1] = 0
    circle = np.zeros(target_size).astype(np.uint8)
    cv2.circle(circle, (target_size[0] // 2, target_size[1] // 2), 100, (1, 1, 1), 1)
    with_circle = circle + sparsed
    with_circle[with_circle == 3] = 2
    with_circle = cv2.copyMakeBorder(
        with_circle, 28, 28, 28, 28, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return with_circle.astype(np.uint8)


# %%
image = images[10300]
sparse_draw = wm38_to_sparse_draw(image.copy())
restored = upscaled_filtered_with_circle_zoom_in(sparse_draw.copy())
# %%
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original WM38")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(sparse_draw)
plt.title("sparse WM38")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(restored)
plt.title("Restored sparse WM38")
plt.colorbar()
# %%
restored = upscaled_filtered_with_circle_zoom_in(sparse_draw)
# %%
labels = labels.astype(np.uint8)
# %%

sparse_images = Parallel(n_jobs=-1)(
    delayed(wm38_to_sparse_draw)(image) for image in tqdm(images)
)
stacked_sparse = np.stack(sparse_images)
np.savez_compressed("sparse_wm38.npz", images=stacked_sparse, labels=labels)
stacked_sparse.shape
plt.imshow(stacked_sparse[0])
stacked_sparse[0].shape
# %%
restored_npz = np.load("restored_wm38.npz", allow_pickle=True)
restored_images = restored_npz["images"]
plt.imshow(restored_images[0])
# %%
restored_images = Parallel(n_jobs=-1)(
    delayed(upscaled_filtered_with_circle_zoom_in)(image)
    for image in tqdm(sparse_images)
)
stacked_restored = np.stack(restored_images)
stacked_restored.shape
plt.imshow(stacked_restored[0])
np.savez_compressed("restored_wm38.npz", images=stacked_restored, labels=labels)
# %%
sparse_npz = np.load("sparse_wm38.npz", allow_pickle=True)
sparse_images = sparse_npz["images"]
labels = sparse_npz["labels"]
plt.imshow(sparse_images[0])
# %%
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sparse_images[0])
plt.title("Sparse WM38")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(restored_images[0])
plt.title("Restored Sparse WM38")
plt.colorbar()
