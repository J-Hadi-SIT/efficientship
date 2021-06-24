import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import swifter
import pickle
import time
import sys
import os
from tqdm import tqdm
from itertools import chain
from datetime import timedelta
from operator import itemgetter
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from scipy.ndimage import gaussian_filter
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import auc, mean_squared_error
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


# chunker function to chunk signal into subsets
chunker = lambda signal, i, stride, time_constant: np.array(signal[i*stride:i*stride+time_constant]).reshape((-1, 1))


def parse_raw(folders, params, length=1):
    # gather file names and path
    files = []
    for folder in folders:
        folder_files = filter(lambda f: f.startswith('v1'), os.listdir(folder))
        folder_files = map(lambda f: os.path.join(folder, f), folder_files)
        files.extend(list(folder_files))
    # gather files content
    raws, unsuccessful = [], []
    for file in tqdm(files):
        with open(file, 'rb') as f:
            try:
                raw = pickle.load(f)
                raw = list(filter(lambda r: True in [param in r[1] for param in params], raw))
                raws.extend(raw)
            except (MemoryError, EOFError, pickle.UnpicklingError):
                unsuccessful.append(file)
    # tabulate as dataframe with datetime index
    df = []
    for param in set(map(itemgetter(1), raws)):
        # epoch_param_value
        datum = np.array(list(filter(lambda epv: epv[1] == param, raws)))
        epoch = datum[:, 0]
        if length == 1:
            value = {param: datum[:, 2]}
        else:
            value = {param + f"_{i}": datum[:, 2+i] for i in range(length)}
        df.append(pd.DataFrame(value, index=epoch))
    df = pd.concat(df)
    df.index = pd.Series(df.index).apply(lambda ts: pd.Timestamp(float(ts), unit='s'))
    df = df.astype(float)
    if len(unsuccessful) > 0:
        print('files not parsed:')
        for f in unsuccessful:
            print(f)
    return df


def create_segments(unchopped, resampling='1S', gap=300, min_data_points=0.5):
    min_data_points *= gap
    gap = timedelta(seconds=gap)
    # find gap mask, False is gap larger than segment_gap
    mask = unchopped.index.to_list()
    mask = [True] + list(map(lambda z: z[1] - z[0] <= gap, zip(mask[:-1], mask[1:])))
    print('number of segments:', mask.count(False))
    segments, bgn = [], 0
    time.sleep(0.25)
    # fill segments
    for _ in tqdm(range(mask.count(False))):
        end = mask.index(False)
        segment = unchopped.iloc[bgn:end]
        if segment.index[-1] - segment.index[0] >= gap and segment.shape[0] >= min_data_points:
            # segment is larger than gap
            segment = segment.fillna(method='ffill').fillna(method='bfill')
            segment.index = segment.index.tz_localize('Asia/Singapore')
            segment = segment.resample('1S').ffill()
            segments.append(segment)
        mask[end] = True
        bgn = end
    time.sleep(0.25)
    print('accepted segments:', len(segments))
    return segments


def find_approx(sig):
    if sig.shape[0] == 1:
        return sig[0]
    else:
        return find_approx((sig[::2] + sig[1::2]) / 2)


def filter_signal(signal, window=3):
    window = 2**window
    sig = signal.to_numpy()
    threshold = [find_approx(sig[i:i+window]) for i in range(sig.shape[0]-window)]
    mask = (signal == 0) & (signal < np.array(threshold[:1] * window + threshold))
    signal = signal[~mask]
    return signal


def import_location(import_from):
    columns = ['mmsi', 'status', 'speed', 'lon', 'lat', 'course', 'heading', 'ts']
    # import
    location = [pd.read_csv(f) for f in import_from]
    location = pd.concat(location, ignore_index=True).reset_index(drop=True)
    location.columns = columns
    # datetime index
    location.index = pd.to_datetime(location['ts'], format='%Y-%m-%d %H:%M:%S', utc=True).dt.tz_convert('Asia/Singapore')
    # trim columns
    location = location.drop(columns=['mmsi', 'status', 'ts'])
    return location


def create_location_params(location):
    location['ts'] = location.index.to_series()
    # reset index
    location = location.reset_index(drop=True)
    # shift
    _lat, _lon = location['lat'].drop(location.shape[0]-1), location['lon'].drop(location.shape[0]-1)
    _idx = pd.to_datetime(location['ts'].drop(location.shape[0]-1), format='%Y-%m-%d %H:%M:%S', utc=True).dt.tz_convert('Asia/Singapore')
    location = location.drop(0).reset_index(drop=True)
    location['_lat'], location['_lon'] = _lat, _lon
    location['t0'] = _idx
    # displacement and direction (degrees ccw from north)
    latlon = location.loc[:, ('_lat', '_lon', 'lat', 'lon')].copy()
    location['distance'] = latlon.apply(lambda x: geodesic(x[:2], x[2:]).meters, axis=1)
    location['direction'] = latlon.apply(lambda x: np.rad2deg(np.arctan((x[3]-x[1])/(x[2]-x[0])))%360, axis=1)
    # datetime index
    location.index = pd.to_datetime(location['ts'], format='%Y-%m-%d %H:%M:%S', utc=True).dt.tz_convert('Asia/Singapore')
    location['t1'] = location.index
    # location['timedelta'] = (location['t1']-location['t0']).astype(np.timedelta64(1, 's'))
    return location


def find_segments_location(segments, import_from, interval=1):
    location = pd.concat([pd.read_csv(f) for f in import_from], ignore_index=True)
    location.columns = ['mmsi', 'status', 'speed', 'lon', 'lat', 'course', 'heading', 'ts']
    location.index = pd.to_datetime(location['ts'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize('Asia/Singapore')
    location = location.loc[:, ('lon', 'lat')]
    location = location.resample('1S').interpolate(method='linear')
    location['_lon'], location['_lat'] = location['lon'].shift(periods=-1), location['lat'].shift(periods=-1)
    location = location.drop(location.index[-1])
    locales = []
    for segment in tqdm(segments):
        locale = location.loc[(location.index >= segment.index[0]) & (location.index <= segment.index[-1]), ('_lat', '_lon', 'lat', 'lon')]
        locale['distance'] = locale.apply(lambda row: geodesic(row[:2], row[-2:]).meters, axis=1)
        # locale = create_location_params(locale)
        locales.append(locale)
    return locales


def ts_clustering(X_train, stride, time_constant, n_cluster):
    # split each segment into chunks, then flatten
    X_train = [chunker(signal, i, stride, time_constant) for signal in X_train for i in range((len(signal)-time_constant)//stride)]
    # transform to tslearn dataset
    X_train = to_time_series_dataset(X_train)
    # clustering and train
    km = TimeSeriesKMeans(n_clusters=n_cluster, metric='euclidean')
    km.fit(X_train)
    # hasher, to turn label to score
    means = [(i, np.mean(agg)) for i, agg in enumerate(km.cluster_centers_)]
    hasher = dict(zip(map(itemgetter(0), sorted(means, key=itemgetter(1))), range(len(means))))
    return km, hasher


def get_scores(segments, segments_wind, segments_location, stride=15, time_constant=60, n_cluster=10):
    # prepare signals
    signals_mflow = [segment.sum(axis=1).to_numpy() for segment in segments]
    signals_wind = [segment['effect'].to_numpy() for segment in segments_wind]
    signals_loc = [segment['distance'].to_numpy() for segment in segments_location]
    signals = [signals_mflow, signals_wind, signals_loc]
    # create and train models, create hashes
    estimators = [ts_clustering(segments, stride, time_constant, n_cluster) for segments in signals]
    # convert to scores
    scores = []
    for (estimator, hash), signal in zip(estimators, signals):
        score = []
        for segment in signal:
            chunk = [chunker(segment, i, stride, time_constant) for i in range((len(segment)-time_constant)//stride)]
            labels = estimator.predict(chunk)
            score.append(np.array([hash[label] for label in labels]))
        scores.append(score)
    # def concurrent_function(sigs, std, tco, est, hsh):
    #     # score = []
    #     # for seg in signal:
    #     #     chunk = [chunker(seg, i, std, tco) for i in range((len(seg)-tco)//std)]
    #     #     labels = est.predict(chunk)
    #     #     score.append(np.array([hsh[label] for label in labels]))
    #     def sub_cf(s, d, t, e, h):
    #         chunk = [chunker(s, i, d, t) for i in range((len(s)-t)//d)]
    #         labels = e.predict(chunk)
    #         return np.array([h[label] for label in labels])
    #     score = []
    #     with ThreadPoolExecutor() as exec:
    #         wf = []
    #         for sig in sigs:
    #             wf.append(exec.submit(sub_cf, sig, std, tco, est, hsh))
    #         for c in as_completed(wf):
    #             score.append(c.result())
    #     return score
    # scores = []
    # with ThreadPoolExecutor() as executor:
    #     wait_for = []
    #     for (estimator, hash), signal in zip(estimators, signals):
    #         wait_for.append(executor.submit(concurrent_function, signal, stride, time_constant, estimator, hash))
    #     for completed in as_completed(wait_for):
    #         scores.append(completed.result())
    return signals, scores, estimators


def generate_initial_proba(segments, param, n_class, cluster_name):
    # # segments = list(chain.from_iterable(segments))
    # segments = [segment[0] for segment in segments]
    # unique, counts = np.unique(segments, return_counts=True)
    # proba = dict(zip(range(n_class), np.zeros((n_class, ))))
    # proba = {**dict(zip(range(n_class), np.zeros((n_class, )))), **dict(zip(unique, counts/len(segments)))}
    # # for k, v in dict(zip(unique, counts/len(segments))).items():
    # #     proba[k] = v
    # proba = map(itemgetter(1), sorted([(k, v) for k, v in proba.items()], key=itemgetter(0)))
    # return cluster_name, np.array(list(proba))
    try:
        segments = [segment[0][0][param] for segment in segments]
    except IndexError:
        segments = [segment[0][param] for segment in segments]
    inits, cts = np.unique(segments, return_counts=True)
    mtx = np.zeros((n_class, ))
    np.put(mtx, inits, cts.astype(np.float64))
    mtx /= np.sum(mtx)
    return cluster_name, mtx


def generate_hidden_matrix(segments, param, n_class, cluster_name):
    # # find unique states and sequential pairs
    # seq_pairs, states = [], []
    # for segment in segments:
    #     states.extend(list(segment))
    #     seq_pairs.extend([segment[i:i+2] for i in range(len(segment)-1)])
    # states = set(states)
    # # find distribution proba, likeliness to move from current
    # unique, counts = np.unique(list(map(lambda pair: pair[1] - pair[0], seq_pairs)), return_counts=True)
    # counts = counts / np.sum(counts)
    # distribution = dict(zip(unique, counts))
    # # find proba to which state next, at what proba
    # next_states = []
    # for group in map(lambda hidden_state: list(filter(lambda pair: pair[0] == hidden_state, seq_pairs)), states):
    #     this_state = set(map(itemgetter(0), group))
    #     if len(this_state) != 1: 
    #         continue
    #     else: 
    #         this_state = list(this_state)[0]
    #     next_state = list(map(itemgetter(1), group))
    #     unique, counts = np.unique(next_state, return_counts=True)
    #     proba = counts / len(next_state)
    #     next_states.append((this_state, dict(zip(unique, proba))))
    # next_states = dict(next_states)
    # # build transition matrix, must account for missing states
    # states = []
    # for this_state in range(n_class):
    #     # all zero
    #     next_state = dict(zip(range(n_class), np.zeros((n_class, ))))
    #     if this_state in next_states:
    #         # overwrite
    #         next_state = {**next_state, **next_states[this_state]}
    #     else:
    #         # next_state = {**next_state, **{(k + this_state): v for k, v in distribution.items() if 0 <= (k + this_state) < n_class}}
    #         next_state[this_state] = 1
    #     states.append(pd.DataFrame(next_state, index=[this_state]))
    # states = pd.concat(states, sort=True)
    # return cluster_name, states.to_numpy()
    try:
        segments = [seg[:,param] for segment in segments for seg in segment]
    except IndexError:
        segments = [segment[:,param] for segment in segments]
        pairs = []
        for segment in segments:
            if len(segment) < 2:
                continue
            pairs.extend(list(np.column_stack((segment[:-1:2], segment[1::2]))))
        segments = np.array(pairs)
    assoc, cts = np.unique(segments, axis=0, return_counts=True)
    mtx = np.zeros((n_class, n_class))
    for (r, c), p in zip(assoc, cts): mtx[r,c] = p
    div = np.tile(np.sum(mtx, axis=1), (mtx.shape[1], 1)).T
    div[div==0] = 1
    mtx /= div
    # fill diagonal
    # for rc in np.where(np.sum(mtx, axis=1) == 0)[0]: mtx[rc,rc] = 1
    return cluster_name, mtx


def generate_observable_matrix(segments, param_hid, param_obs, n_class, cluster_name):
    # flatten then zip
    # association = list(zip([s for segment in segments_hidden for s in segment], 
    #                        [s for segment in segments_observable for s in segment]))
    # association_df = []
    # for this_state in range(n_class):
    #     # for every class, fill with initial zero proba
    #     class_association = np.zeros((n_class, ))
    #     # sift for this iteration
    #     class_members = list(filter(lambda z: z[0] == this_state, association))
    #     if len(class_members) != 0:
    #         evident_class_association = list(map(itemgetter(1), class_members))
    #         # get number of counts
    #         unique, counts = np.unique(evident_class_association, return_counts=True)
    #         # update proba
    #         class_association[unique] = counts / len(evident_class_association)
    #     else:
    #         # print('no association:', this_state)
    #         pass
    #     association_df.append(pd.DataFrame([class_association], index=[this_state]))
    # association_df = pd.concat(association_df, sort=True)
    # return cluster_name, association_df.to_numpy()
    try:
        hiddens = np.array([[seg[:,param_hid],] for segment in segments for seg in segment])
        observables = np.array([[seg[:,param_obs],] for segment in segments for seg in segment])
    except IndexError:
        hiddens = list(chain.from_iterable([segment[:,param_hid] for segment in segments]))
        observables = list(chain.from_iterable([segment[:,param_obs] for segment in segments]))
    # hidden = np.array(segments_hidden).flatten()
    # observable = np.array(segments_observable).flatten()
    hid_obs = np.column_stack((hiddens, observables))
    assoc, cts = np.unique(hid_obs, axis=0, return_counts=True)
    mtx = np.zeros((n_class, n_class))
    for (r, c), p in zip(assoc, cts): mtx[r,c] = p
    div = np.tile(np.sum(mtx, axis=1), (mtx.shape[1], 1)).T
    div[div==0] = 1
    mtx /= div
    # fill diagonal
    # for rc in np.where(np.sum(mtx, axis=1) == 0)[0]: mtx[rc,rc] = 1
    return cluster_name, mtx


# find datapoints cluster member
def associate_to_cluster(data, centers):
    distance_to_centers = np.array([np.linalg.norm(data - center, axis=1) for center in centers]).T
    minimum_distance = np.array([np.min(distance) for distance in distance_to_centers])
    cluster_member_mask = np.array([distances == minimum for distances, minimum in zip(distance_to_centers, minimum_distance)])
    cluster_member_card = np.array([np.ones((data.shape[0], )) * i for i in range(len(centers))]).T
    datapoint_labels = np.array([card[mask][0] for card, mask in zip(cluster_member_card, cluster_member_mask)])
    return datapoint_labels
    

def get_cluster_member(score, cluster_centers):
    seg = np.array(score).T
    dist = np.array([np.linalg.norm(seg - center, axis=1) for center in cluster_centers]).T
    minim = np.min(dist, axis=1)
    mask = np.array([d == minim for d in dist.T]).T
    cluster_member = np.array([np.ones((mask.shape[0], )) * i for i in range(np.shape(cluster_centers)[0])]).T[mask]
    return cluster_member


def create_dataset(n_cluster, scores, test_segment, est):
    dataset = {i: [] for i in range(n_cluster)}
    bgn = 0
    for enum, score in enumerate(zip(*scores)):
        if enum == test_segment:
            continue
        # segment points
        seg = np.array(score).T
        # find end then slice
        end = bgn + seg.shape[0]
        cluster_member = est.labels_[bgn:end]
        # for every cluster
        for i in range(n_cluster):
            # get index 
            sequence = np.where(cluster_member == i)[0]
            # must be 2 or more
            if sequence.shape[0] < 2:
                continue
            # cut sequence once leaving cluster
            cuts = [0] + list(np.where(np.diff(sequence) != 1)[0] + 1) + [sequence.shape[0]]
            for p0, p1 in zip(cuts[:-1], cuts[1:]):
                p0, p1 = sequence[p0:p1][0], sequence[p0:p1][-1]+1
                member = seg[p0:p1]
                dataset[i].append(member)
        # for next loop
        bgn = end
    # check for empty clusters
    for k, v in dataset.items():
        if len(v) == 0:
            print(k, 'has no member')
    return dataset


def hmm_modelling(dataset, n_class, n_cluster, hidden_param, observable_param):
    hmm = {}
    pi, hidden, observable = [], [], []
    with ThreadPoolExecutor() as executor:
        wait_for_1, wait_for_2, wait_for_3 = [], [], []
        for i in range(n_cluster):
            wait_for_1.append(executor.submit(generate_initial_proba, dataset[i], hidden_param, n_class, i))
            wait_for_2.append(executor.submit(generate_hidden_matrix, dataset[i], hidden_param, n_class, i))
            wait_for_3.append(executor.submit(generate_observable_matrix, dataset[i], hidden_param, observable_param, n_class, i))
        for f in as_completed(wait_for_1):
            pi.append(f.result())
        for f in as_completed(wait_for_2):
            hidden.append(f.result())
        for f in as_completed(wait_for_3):
            observable.append(f.result())
    pi = list(map(itemgetter(1), sorted(pi, key=itemgetter(0))))
    hidden = list(map(itemgetter(1), sorted(hidden, key=itemgetter(0))))
    observable = list(map(itemgetter(1), sorted(observable, key=itemgetter(0))))
    for i, zipped in enumerate(zip(pi, hidden, observable)):
        hmm[i] = zipped
    return hmm


def smoothen(inp, sig=1):
    to_blur = gaussian_filter(inp.copy(), sigma=sig)
    if len(to_blur.shape) > 1:
        div = np.sum(to_blur, axis=1)
        div[div==0] = 1
        div = np.tile(div, (to_blur.shape[1], 1)).T
    else:
        div = np.sum(to_blur)
    to_blur /= div
    return to_blur


def predict(scores, test_segment, est, hmm, sigma):
    score = np.array([scores[i][test_segment] for i in range(3)])
    cluster_member = est.predict(score.T)
    # prepare for viterbi
    level = cluster_member[0]
    cluster_member_segments, one_level = [], [level]
    for i in range(1, cluster_member.shape[0]):
        if cluster_member[i] == level:
            one_level.append(cluster_member[i])
        else:
            cluster_member_segments.append(one_level)
            level = cluster_member[i]
            one_level = [level]
    if len(one_level) > 0:
        cluster_member_segments.append(one_level)
    # use viterbi
    path, bgn = [], 0
    for one_level in cluster_member_segments:
        end = bgn + len(one_level)
        pi, hidden, observable = hmm[one_level[0]]
        pi = smoothen(pi, sig=sigma)
        hidden = smoothen(hidden, sig=sigma)
        observable = smoothen(observable, sig=sigma)
        p, *_ = viterbi(pi, hidden, observable, scores[2][test_segment][bgn:end])
        path.extend(p)
        bgn = end
    return cluster_member, path


def viterbi(pi, a, b, obs):
    
    n_states = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = path = np.zeros(T, dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((n_states, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((n_states, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    # print('\nstart walking forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(n_states):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            # print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    # print('-' * 50)
    # print('start backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        # print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi


def process_all_segments(save_as):
    global n_cluster
    with open(dataset_path, 'rb') as f:
        ds = pickle.load(f)
    indices, segments, segments_wind, segments_location = ds['index'], ds['bunker'], ds['wind'], ds['location']
    signals, scores, estimators = get_scores(segments, segments_wind, segments_location, 
                                             stride=stride,
                                             time_constant=time_constant,
                                             n_cluster=n_class)
    results = []
    for test_segment in tqdm(range(len(indices))):
        print(f"\nprocessing segment #{test_segment}")
        # create datapoints, and apply bias
        b = np.concatenate([score for enum, score in enumerate(scores[0]) if enum != test_segment]) * biases[0]
        w = np.concatenate([score for enum, score in enumerate(scores[1]) if enum != test_segment]) * biases[1]
        d = np.concatenate([score for enum, score in enumerate(scores[2]) if enum != test_segment]) * biases[2]
        datapoints = np.array([b, w, d]).T
        # k-means
        n_cluster_ = n_cluster
        cluster_centers = []
        if n_cluster_ > infinite_breaker:
            print('optimum not found')
            break
        # create clusters, without noise points
        if centers is not None:
            est = KMeans(n_clusters=n_cluster_, init=centers)
        else:
            est = KMeans(n_clusters=n_cluster_)
        est.fit(datapoints)
        # inverse mean square error
        cluster_quality = [np.linalg.norm(datapoints[est.labels_==i] - est.cluster_centers_[i], axis=1) for i in range(n_cluster_)]
        cluster_quality = n_class / np.array([np.mean(cq**2) for cq in cluster_quality])
        print(f"n_cluster: {n_cluster_}, minimum: {np.min(cluster_quality)}")
        n_cluster = est.cluster_centers_.shape[0]
        # prepare for training
        dataset = {i: [] for i in range(n_cluster)}
        bgn = 0
        for enum, score in enumerate(zip(*scores)):
            if enum == test_segment:
                continue
            # segment points
            seg = np.array(score).T
            # find end then slice
            end = bgn + seg.shape[0]
            cluster_member = est.labels_[bgn:end]
            # for every cluster
            for i in range(n_cluster):
                # get index 
                sequence = np.where(cluster_member == i)[0]
                # must be 2 or more
                if sequence.shape[0] < 2:
                    continue
                cuts = [0] + list(np.where(np.diff(sequence) != 1)[0] + 1) + [sequence.shape[0]]
                for p0, p1 in zip(cuts[:-1], cuts[1:]):
                    p0, p1 = sequence[p0:p1][0], sequence[p0:p1][-1]+1
                    member = seg[p0:p1]
                    dataset[i].append(member)
            bgn = end
        for k, v in dataset.items():
            if len(v) == 0:
                print(k, 'has no member')
        # 0 bunker, 1 wind effect, 2 distance
        hidden_param = 0
        observable_param = 2
        # hidden: to be predicted
        # observable: input
        hmm2 = {}
        pi, hidden, observable = [], [], []
        with ThreadPoolExecutor() as executor:
            wait_for_1, wait_for_2, wait_for_3 = [], [], []
            for i in range(n_cluster):
                wait_for_1.append(executor.submit(generate_initial_proba, dataset[i], hidden_param, n_class, i))
                wait_for_2.append(executor.submit(generate_hidden_matrix, dataset[i], hidden_param, n_class, i))
                wait_for_3.append(executor.submit(generate_observable_matrix, dataset[i], hidden_param, observable_param, n_class, i))
            for f in as_completed(wait_for_1):
                pi.append(f.result())
            for f in as_completed(wait_for_2):
                hidden.append(f.result())
            for f in as_completed(wait_for_3):
                observable.append(f.result())
        pi = list(map(itemgetter(1), sorted(pi, key=itemgetter(0))))
        hidden = list(map(itemgetter(1), sorted(hidden, key=itemgetter(0))))
        observable = list(map(itemgetter(1), sorted(observable, key=itemgetter(0))))
        for i, zipped in enumerate(zip(pi, hidden, observable)):
            hmm2[i] = zipped
        score = np.array([scores[i][test_segment] for i in range(3)])
        cluster_member = est.predict(score.T)
        level = cluster_member[0]
        cluster_member_segments, one_level = [], [level]
        for i in range(1, cluster_member.shape[0]):
            if cluster_member[i] == level:
                one_level.append(cluster_member[i])
            else:
                cluster_member_segments.append(one_level)
                level = cluster_member[i]
                one_level = [level]
        if len(one_level) > 0:
            cluster_member_segments.append(one_level)
        path2, bgn = [], 0
        for one_level in cluster_member_segments:
            end = bgn + len(one_level)
            pi, hidden, observable = hmm2[one_level[0]]
            pi = smoothen(pi, sig=sigma)
            hidden = smoothen(hidden, sig=sigma)
            observable = smoothen(observable, sig=sigma)
            p, *_ = viterbi(pi, hidden, observable, scores[2][test_segment][bgn:end])
            path2.extend(p)
            bgn = end
        auc_truth = auc(range(len(score[hidden_param])), score[hidden_param])
        auc_pred = auc(range(len(path2)), path2)
        auc_diff = (auc_pred - auc_truth) / auc_truth
        mse = mean_squared_error(score[hidden_param], path2)
        results.append([auc_diff, mse])
    with open(save_as, 'wb') as f:
        pickle.dump(results, f)
    return results


dataset_path = 'dataset_5s.pkl'
n_class = 100
stride = 3
time_constant = 12
biases = [1, 1, 1]
n_cluster = 300
infinite_breaker = 500
cluster_quality_thresh = 0.5
centers = None
sigma = 1


if __name__ == '__main__':
    process_all_segments(sys.argv[1])
