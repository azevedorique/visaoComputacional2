import numpy as np
from scipy.spatial import KDTree


 
def nearest_neighbors(source, target):
    tree = KDTree(target)
    distances, indices = tree.query(source)

    return distances, indices


def reject_outliers(source, target, distances, indices,
                    max_distance=0.05):

    mask = distances < max_distance

    source_inliers = source[mask]
    target_inliers = target[indices[mask]]

    return source_inliers, target_inliers, mask


def estimate_transform(P, Q):

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_Q - R @ centroid_P

    return R, t


def transform_points(points, R, t):
    return (R @ points.T).T + t


 

def icp_iteration(source,
                  target,
                  max_distance=0.05):

    distances, indices = nearest_neighbors(
        source,
        target
    )

    source_inliers, target_inliers, mask = reject_outliers(
        source,
        target,
        distances,
        indices,
        max_distance
    )

    inlier_ratio = len(source_inliers) / len(source)

    if len(source_inliers) < 3:
        return source, np.eye(3), np.zeros(3), inlier_ratio

    R, t = estimate_transform(
        source_inliers,
        target_inliers
    )

    transformed = transform_points(source, R, t)

    return transformed, R, t, inlier_ratio


 

def run_icp(source,
            target,
            max_iterations=20,
            max_distance=0.05):

    current = source.copy()

    R_total = np.eye(3)
    t_total = np.zeros(3)

    first_iteration_ratio = None

    for i in range(max_iterations):

        current, R, t, ratio = icp_iteration(
            current,
            target,
            max_distance
        )

        if i == 0:
            first_iteration_ratio = ratio

        R_total = R @ R_total
        t_total = R @ t_total + t

    return {
        "aligned": current,
        "R": R_total,
        "t": t_total,
        "first_ratio": first_iteration_ratio
    }


 

class KeyframeICP:

    def __init__(self):

        self.keyframe = None
        self.backup_frame = None

        self.prev_ratio = 1.0

        self.TH_IR1 = 0.4
        self.TH_RATIO_DIFF = 0.3

        self.translation_threshold = 0.08

    def process(self, current_frame):

        

        if self.keyframe is None:

            self.keyframe = current_frame
            self.backup_frame = current_frame

            return np.eye(4)

    

        test = run_icp(
            self.keyframe,
            current_frame,
            max_iterations=1
        )

        current_ratio = test["first_ratio"]

        
        DCS = current_ratio < self.TH_IR1

         
        DCT = (
            abs(current_ratio - self.prev_ratio)
            > self.TH_RATIO_DIFF
        )

        

        if DCS or DCT:

            source = self.backup_frame

            print("Usando BACKUP FRAME")

        else:

            source = self.keyframe

            print("Usando KEYFRAME")

         
        result = run_icp(
            source,
            current_frame,
            max_iterations=20
        )

        t = result["t"]

        translation_norm = np.linalg.norm(t)

         
        if translation_norm > self.translation_threshold:

            self.keyframe = current_frame

            print("Novo KEYFRAME")

        

        if current_ratio > self.TH_IR1:

            self.backup_frame = current_frame

        self.prev_ratio = current_ratio

        T = np.eye(4)
        T[:3, :3] = result["R"]
        T[:3, 3] = result["t"]

        return T


 
if __name__ == "__main__":

    system = KeyframeICP()

    for i in range(100):

        source = np.random.rand(1000, 3)

        target = source + np.array([
            0.01 * i,
            0,
            0
        ])

        pose = system.process(target)

        print(pose)