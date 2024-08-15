import geomstats.backend as gs
import numpy as np 
from numba import jit, njit, prange
import scipy.stats as stats
from scipy.integrate import simpson
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    DynamicProgrammingAligner,
    SRVReparametrizationBundle,
    SRVRotationBundle,
    SRVRotationReparametrizationBundle,
    SRVMetric
)


def del_arr_elements(arr, indices):
    """
    Delete elements in indices from array arr
    """

    # Sort the indices in reverse order to avoid index shifting during deletion
    indices.sort(reverse=True)

    # Iterate over each index in the list of indices
    for index in indices:
        del arr[index]
    return arr



@jit(nopython=False, forceobj=True)
def parallel_dist(cells, dist_fun, k_sampling_points):
    pairwise_dists = np.zeros((cells.shape[0], cells.shape[0]))
    for i in prange(cells.shape[0]):
        for j in prange(i + 1, cells.shape[0]):
            pairwise_dists[i, j] = dist_fun(cells[i], cells[j]) / k_sampling_points
    pairwise_dists += pairwise_dists.T
    return pairwise_dists


def remove_cell_shapes(cell_shapes, ds_align, delete_indices, num_layer):
    """ 
    Remove cells of control group from cells, cell_shapes, lines, ds_align,
    the parameters returned from load_treated_osteosarcoma_cells
    Also update n_cells

    :param list[int] delete_indices: the indices to delete
    """
    delete_indices.sort(reverse=True) # to prevent change in index when deleting elements
    
    # Delete elements
    cell_shapes = np.delete(np.array(cell_shapes), delete_indices, axis=0)
    if num_layer == 1:
        ds_align = remove_ds_align_one_layer(ds_align, delete_indices)
    elif num_layer == 2:
        ds_align = remove_ds_align_two_layer(ds_align, delete_indices)
    return cell_shapes, ds_align


def overlap_ratio(distance1, distance2):
    """ 
    Calculate the ratio of overlap regions between the histograms of distance1 and distance

    :param list[float] distance1: list of positive distances 
    :param list[float] distance2: list of positive distances 
    :param function kde1: the kernel density estimation of distance1
    :param function kde2: the kernel density estimation of distance2
    """

    # Define a common set of points for evaluation (covering the range of both datasets)
    x_eval = np.linspace(min(np.min(distance1), np.min(distance2)), max(np.max(distance1), np.max(distance2)), 1000)

    # Create KDEs for the two datasets
    kde1 = stats.gaussian_kde(distance1)
    kde2 = stats.gaussian_kde(distance2)

    # Evaluate the KDEs on these points
    kde_values1 = kde1(x_eval)
    kde_values2 = kde2(x_eval)

    # Find the minimum of the two KDEs at each point to determine the overlap
    overlap_values = np.minimum(kde_values1, kde_values2)

    # Integrate the overlap using the composite Simpson's rule
    overlap_area = simpson(overlap_values, x=x_eval)

    # Calculate the total area under one of the KDEs as a reference (should be close to 1)
    total_area = simpson(kde_values1, x=x_eval)  # or use kde_values2, should be about the same

    # Calculate the ratio of the overlap
    overlap_ratio = (overlap_area / total_area) 

    return overlap_ratio


def knn_score(pos, labels):
    clf = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(clf, pos, labels, cv=5, scoring='accuracy')
    return scores.mean()


def exhaustive_align(curve, ref_curve, k_sampling_points, rescale=True, dynamic=False, reparameterization=True):
    """ 
    Quotient out
        - translation (move curve to start at the origin) 
        - rescaling (normalize to have length one)
        - rotation (try different starting points, during alignment)
        - reparametrization (resampling in the discrete case, during alignment)
    
    :param bool rescale: quotient out rescaling or not 
    :param bool dynamic: Use dynamic aligner or not 
    :param bool reparamterization: quotient out rotation only rather than rotation and reparameterization

    """
    
    curves_r2 = DiscreteCurvesStartingAtOrigin(
        ambient_dim=2, k_sampling_points=k_sampling_points, equip=False
    )

    if dynamic:
        print("Use dynamic programming aligner")
        curves_r2.fiber_bundle = SRVReparametrizationBundle(curves_r2)
        curves_r2.fiber_bundle.aligner = DynamicProgrammingAligner()

    # Quotient out translation
    print("Quotienting out translation")
    curve = curves_r2.projection(curve)
    ref_curve = curves_r2.projection(ref_curve)

    # Quotient out rescaling
    if rescale:
        print("Quotienting out rescaling")
        curve = curves_r2.normalize(curve)
        ref_curve = curves_r2.normalize(ref_curve)

    # Quotient out rotation and reparamterization
    curves_r2.equip_with_metric(SRVMetric)
    if not reparameterization:
        print("Quotienting out rotation")
        curves_r2.equip_with_group_action("rotations")
    else:
        print("Quotienting out rotation and reparamterization")
        curves_r2.equip_with_group_action("rotations and reparametrizations")
        
    curves_r2.equip_with_quotient_structure()
    aligned_curve = curves_r2.fiber_bundle.align(curve, ref_curve)
    return aligned_curve


def knn_score(pos, labels):
    clf = KNeighborsClassifier(n_neighbors=4)
    scores = cross_val_score(clf, pos, labels, cv=5, scoring='accuracy')
    return scores.mean()

def svm_score(pos, labels):
    clf = svm.SVC(kernel='rbf', C=1)
    scores = cross_val_score(clf, pos, labels, cv=5, scoring='accuracy')
    return scores.mean()


def generate_circle_points(num_points):
    # Generate angles evenly distributed between 0 and 2*pi
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Calculate x and y coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Combine x and y coordinates into a 200x2 array
    points = np.column_stack((x, y))
    
    return points


def generate_ellipse(n_sampling, a=10, b = 3):
    """
    Generate points on an ellipse centered at the origin.

    Parameters:
    - a: Semi-major axis of the ellipse.
    - b: Semi-minor axis of the ellipse.
    - n_sampling: Number of points to sample along the ellipse.

    Returns:
    - An array of shape (n_sampling, 2) where each row contains the x and y coordinates of a point on the ellipse.
    """
    # Angles at which to sample the ellipse
    theta = np.linspace(0, 2 * np.pi, n_sampling, endpoint=False)
    
    # Calculate x and y coordinates
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    
    # Combine x and y coordinates into a single array
    points = np.vstack((x, y)).T
    
    return points


def scaled_stress(pos, pairwise_dists):
    """ 
    Calculate the scaled stress invariant to scaling using the original stress \
    statistics and actual pairwise distances

    :param float unscaled_stress: the original stress
    :param 2D np.array[float] pairwise_dists: pairwise distance
    """
    
    # compute pairwise distance of pos
    pairwise_pos = np.empty(shape=(pos.shape[0], pos.shape[0]))
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            pairwise_pos[i,j] = np.sqrt(np.sum(pos[i]-pos[j])**2)
    
    stress = np.sqrt(np.sum((pairwise_dists-pairwise_pos)**2))
    
    return stress/np.sqrt(np.sum(pairwise_dists**2))


def svm_5_fold_classification(X, y):
    # Initialize a Support Vector Classifier
    svm_classifier = svm.SVC(kernel='rbf')

    # Prepare to split the data into 5 folds, maintaining the percentage of samples for each class
    skf = StratifiedKFold(n_splits=5)
    
    # To store precision and recall per class for each fold
    precisions_per_class = []
    recalls_per_class = []

    # Perform 5-fold cross-validation
    for train_index, test_index in skf.split(X, y):
        # Splitting data into training and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        svm_classifier.fit(X_train, y_train)
        
        # Predict on the test data
        y_pred = svm_classifier.predict(X_test)

        # Calculate precision and recall per class
        precision = precision_score(y_test, y_pred, average=None, zero_division=np.nan)
        recall = recall_score(y_test, y_pred, average=None, zero_division=np.nan)

        # Store results from each fold
        precisions_per_class.append(precision)
        recalls_per_class.append(recall)
    
    # Calculate the mean precision and recall per class across all folds
    mean_precisions = np.mean(precisions_per_class, axis=0)
    mean_recalls = np.mean(recalls_per_class, axis=0)

    np.set_printoptions(precision=2)
    
    print("Mean precisions per class across all folds:", mean_precisions)
    print("Mean recalls per class across all folds:", mean_recalls)

    return mean_precisions, mean_recalls


def rotation_align(curve, base_curve, k_sampling_points):
    """Align curve to base_curve to minimize the L² distance by \
        trying different start points.

    Returns
    -------
    aligned_curve : discrete curve
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)

    # Rotation is done after projection, so the origin is removed
    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=k_sampling_points-1)
    total_space.fiber_bundle = SRVRotationBundle(total_space)

    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = total_space.fiber_bundle.align(
            point=gs.array(reparametrized), base_point=base_curve
        )
        distances[shift] = np.linalg.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = total_space.fiber_bundle.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve


def align(point, base_point, rescale, rotation, reparameterization, k_sampling_points):
    """
    Align point and base_point via quotienting out translation, rescaling, rotation and reparameterization
    """

    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=k_sampling_points)
   
    
    # Quotient out translation 
    point = total_space.projection(point) 
    point = point - gs.mean(point, axis=0)

    base_point = total_space.projection(base_point)
    base_point = base_point - gs.mean(base_point, axis=0)

    # Quotient out rescaling
    if rescale:
        point = total_space.normalize(point) 
        base_point = total_space.normalize(base_point)
    
    # Quotient out rotation
    if rotation:
        point = rotation_align(point, base_point, k_sampling_points)

    # Quotient out reparameterization
    if reparameterization:
        aligner = DynamicProgrammingAligner(total_space)
        total_space.fiber_bundle = SRVReparametrizationBundle(total_space, aligner=aligner)
        point = total_space.fiber_bundle.align(point, base_point)
    return point