import numpy as np
import ot  # Python Optimal Transport library
# Generate synthetic images as distributions
image_t0 = np.random.rand(64, 64)
image_t1 = np.random.rand(64, 64)
image_t0 = image_t0/image_t0.sum()
image_t1 = image_t1/image_t1.sum()

# Flatten images and define a ground metric
shape = image_t0.shape
dist_t0 = image_t0.flatten()
dist_t1 = image_t1.flatten()
cost_matrix = ot.utils.dist(np.indices(shape).reshape(2, -1).T, np.indices(shape).reshape(2, -1).T)

# Compute optimal transport matrix
transport_map = ot.emd(dist_t0, dist_t1, cost_matrix)

# Reshape to visualize
transport_map_reshaped = transport_map.reshape(shape)