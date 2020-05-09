from __future__ import absolute_import
from sklearn.utils.linear_assignment_ import linear_assignment
import scipy.linalg
import numpy as np


INFTY_COST = 1e+5
CHI2INV95 = {1: 3.8415,
             2: 5.9915,
             3: 7.8147,
             4: 9.4877,
             5: 11.070,
             6: 12.592,
             7: 14.067,
             8: 15.507,
             9: 16.919}


def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=INFTY_COST, only_position=False):
    gating_dim = 2 if only_position else 4
    gating_threshold = CHI2INV95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


def iou(bbox, candidates):
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix


class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               1e-5,
               10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   1e-2,
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   1e-5,
                   self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha


class Metric(object):
    def __init__(self, metric, matching_threshold):

        if metric == "euclidean":
            self._metric = self._nn_euclidean_distance
        elif metric == "cosine":
            self._metric = self._nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.samples = {}

    @staticmethod
    def _nn_euclidean_distance(x, y):
        x, y = np.asarray(x), np.asarray(y)
        if len(x) == 0 or len(y) == 0:
            return np.zeros((len(x), len(y)))
        a2, b2 = np.square(x).sum(axis=1), np.square(y).sum(axis=1)
        r2 = -2. * np.dot(x, y.T) + a2[:, None] + b2[None, :]
        r2 = np.clip(r2, 0., float(np.inf))
        return np.maximum(0.0, r2.min(axis=0))

    @staticmethod
    def _nn_cosine_distance(x, y):
        x = np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True)
        y = np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)
        return (1. - np.dot(x, y.T)).min(axis=0)

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] += ret[:2]
        return ret

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted


class Tracker:
    def __init__(self, metric_mode, metric_threshold, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = Metric(metric_mode, metric_threshold)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, persons):
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_persons = self._match(persons)

        # Update track set.
        for track_idx, person_idx in matches:
            self.tracks[track_idx].update(self.kf, persons[person_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for person_idx in unmatched_persons:
            self._initiate_track(persons[person_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _gated_metric(self, tracks, dets, track_indices, person_indices):
        features = np.array([dets[i].feature for i in person_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix = gate_cost_matrix(
            self.kf, cost_matrix, tracks, dets, track_indices,
            person_indices)
        return cost_matrix

    def _match(self, persons):

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_persons = matching_cascade(
            self._gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, persons, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_persons = min_cost_matching(
            iou_cost, self.max_iou_distance, self.tracks, persons, iou_track_candidates, unmatched_persons)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_persons

    def _initiate_track(self, person):
        mean, covariance = self.kf.initiate(person.to_xyah())
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, person.feature))
        self._next_id += 1
