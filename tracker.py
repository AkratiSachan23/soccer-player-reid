import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, tlbr, feat, tid):
        self.id = tid
        self.features = [feat]
        self.kf = self._init_kf(tlbr)
        self.time_since_update = 0
    
    def _init_kf(self, tlbr):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        kf.F = np.array([
            [1,0,0,0,dt,0,0], [0,1,0,0,0,dt,0],
            [0,0,1,0,0,0,dt], [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0], [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]])
        kf.H = np.array([
            [1,0,0,0,0,0,0], [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])
        kf.P *= 10.
        kf.R *= 1.
        kf.Q *= 0.01
        cx, cy, a, h = self._tlbr_to_state(tlbr)
        kf.x[:7] = np.array([[cx], [cy], [a], [h], [0.], [0.], [0.]])
        return kf
    
    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x
    
    def update(self, tlbr, feat):
        self.time_since_update = 0
        self.kf.update(self._tlbr_to_state(tlbr))
        self.features.append(feat)
    
    def _tlbr_to_state(self, tlbr):
        x1, y1, x2, y2 = tlbr
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        a = w / float(h)
        return np.array([cx, cy, a, h])

class Tracker:
    def __init__(self, max_age=30, thres=0.5):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.thres = thres
    
    def update(self, detections):
        for t in self.tracks:
            t.predict()
        
        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    cost[i,j] = 1 - np.dot(t.features[-1], d['feat'])
            row, col = linear_sum_assignment(cost)
        else:
            row, col = np.array([]), np.array([])
        
        matched_t, matched_d = set(), set()
        for i, j in zip(row, col):
            if cost[i,j] < self.thres:
                self.tracks[i].update(detections[j]['tlbr'], detections[j]['feat'])
                matched_t.add(i)
                matched_d.add(j)
        
        for j, d in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(Track(d['tlbr'], d['feat'], self.next_id))
                self.next_id += 1
        
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        return [(t.kf.x, t.id) for t in self.tracks if t.time_since_update == 0]