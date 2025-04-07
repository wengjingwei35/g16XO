import sys
import numpy as np
from copy import deepcopy
from random import sample
from math import ceil
import matplotlib.pyplot as plt

CLUSTERMAXVAL = float("inf")
MINDEVIATION = 0.01
MAXKMEANCYCLE = 100

### ----- ###
#0
def k_centers_full(featureArray: np.ndarray, nCluster: int, isRandom=True) -> tuple:
    dist2dArray = get_distance_from_array(featureArray)
    idxLstPerClusterDict, centerIdLst = k_centers_for_distance_2d_array(dist2dArray, nCluster, isRandom)
    labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
    return labelLst, centerIdLst

#0
def k_centers_full_plot(featureArray: np.ndarray, nCluster: int, isRandom=True) -> tuple:
    dist2dArray = get_distance_from_array(featureArray)
    idxLstPerClusterDict, centerIdLst = \
            k_centers_for_distance_2d_array_plot(featureArray, dist2dArray, nCluster, isRandom)
    labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
    return labelLst, centerIdLst

#1
def k_centers_for_distance_2d_array_plot(featureArray: np.ndarray, dist2dArray: np.ndarray, 
                                         nCluster: int, isRandom=True) -> tuple:
    nRow, nCol = 4, 5
    fig, axes = plt.subplots(nRow, nCol, figsize=(nCol*4, nRow*4))
    nUnit = dist2dArray.shape[0]
    centerIdLst = get_initial_cluster(nUnit, isRandom)
    iStep = 0
    while len(centerIdLst) < nCluster:
        iUnit = get_idx_with_minimal_noncenter_to_center_entry(dist2dArray, centerIdLst)
        idxLstPerClusterDict = assign_noncenter_units_to_centers(dist2dArray, centerIdLst)
        labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
        plot_points_and_centers(axes[int(iStep/nCol)][iStep%nCol], featureArray, labelLst, centerIdLst, str(iStep))
        iStep += 1
        centerIdLst.append(iUnit)
        # print("D", centerIdLst)
    idxLstPerClusterDict = assign_noncenter_units_to_centers(dist2dArray, centerIdLst)
    labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
    plot_points_and_centers(axes[int(iStep/nCol)][iStep%nCol], featureArray, labelLst, centerIdLst, str(iStep))
    fig.savefig("k_centers.png", dpi=100)
    return idxLstPerClusterDict, centerIdLst

#2
def plot_points_and_centers(ax, featureArray: np.ndarray, labelLst: list, centerIdLst: list, titleStr: str):
    ax.scatter(featureArray[:,0], featureArray[:,1], s=50, c=["C" + str(label) for label in labelLst], alpha=0.2)
    ax.set_title(titleStr)
    ax.set_xlim((-25,125))
    ax.set_ylim((-25,125))
    for i, centerId in enumerate(centerIdLst):
        ax.scatter(featureArray[centerId,0], featureArray[centerId,1], s=300, c=["C" + str(i)], marker="*")
    
### ----- ###


### ----- ###
#0
def balanced_k_means_full(featureArray: np.ndarray, nCluster: int, isKCenters=False) -> tuple:
    dist2dArray = get_distance_from_array(featureArray)
    nSample = featureArray.shape[0]
    if isKCenters:
        _, initCenterIdLst = k_centers_for_distance_2d_array(dist2dArray, nCluster)
    else:
        initCenterIdLst = sample(list(range(nSample)), nCluster)
    maxClusterWeight = ceil(nSample / nCluster)
    weightArray = np.ones(nSample)
    idxLstPerClusterDict, centerIdLst, clusterSum = balanced_k_means_for_distance_2d_array_with_initial_assignment(\
            dist2dArray, maxClusterWeight, weightArray, initCenterIdLst)
    labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
    return labelLst, centerIdLst, clusterSum

#0
def balanced_k_means_full_plot(featureArray: np.ndarray, nCluster: int, isKCenters=False) -> tuple:
    dist2dArray = get_distance_from_array(featureArray)
    nSample = featureArray.shape[0]
    if isKCenters:
        _, initCenterIdLst = k_centers_for_distance_2d_array(dist2dArray, nCluster)
    else:
        initCenterIdLst = sample(list(range(nSample)), nCluster)
    maxClusterWeight = ceil(nSample / nCluster)
    weightArray = np.ones(nSample)
    idxLstPerClusterDict, centerIdLst, clusterSum = \
            balanced_k_means_for_distance_2d_array_with_initial_assignment_plot(\
                    featureArray, dist2dArray, maxClusterWeight, weightArray, initCenterIdLst, isKCenters)
    labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
    return labelLst, centerIdLst, clusterSum

#1
def balanced_k_means_for_distance_2d_array_with_initial_assignment_plot(\
        featureArray: np.ndarray, dist2dArray: np.ndarray, maxClusterWeight: float, weightArray: np.ndarray, 
        initCenterIdLst: list, isKCenters: bool) -> tuple:
    nRow, nCol = 4, 5
    fig, axes = plt.subplots(nRow, nCol, figsize=(nCol*4, nRow*4))
    iStep = 0
    idxLstPerClusterDict = get_index_list_per_cluster_dict(dist2dArray, initCenterIdLst, 
                                                           maxClusterWeight, weightArray)
    oldClusterSum = 99999999999999999.0
    newClusterSum = get_cluster_sum(dist2dArray, initCenterIdLst, idxLstPerClusterDict)
    labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, initCenterIdLst)
    plot_points_and_centers(axes[int(iStep/nCol)][iStep%nCol], featureArray, labelLst, initCenterIdLst, 
                            "{0}: {1:.4e}".format(iStep, newClusterSum))
    while abs(oldClusterSum - newClusterSum) > abs(oldClusterSum) * MINDEVIATION:
        # print("D-", oldClusterSum, newClusterSum)
        # print("D+", oldClusterSum - newClusterSum, abs(oldClusterSum) * MINDEVIATION)
        oldClusterSum = newClusterSum
        iStep += 1
        centerIdLst = get_new_center_id_list(dist2dArray, idxLstPerClusterDict)
        # print("D", centerIdLst)
        idxLstPerClusterDict = get_index_list_per_cluster_dict(dist2dArray, centerIdLst, 
                                                               maxClusterWeight, weightArray)
        newClusterSum = get_cluster_sum(dist2dArray, centerIdLst, idxLstPerClusterDict)
        # print("Dn", oldClusterSum, newClusterSum)
        labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
        plot_points_and_centers(axes[int(iStep/nCol)][iStep%nCol], featureArray, labelLst, centerIdLst, 
                                "{0}: {1:.4e}".format(iStep, newClusterSum))
    if isKCenters:
        fig.savefig("balanced_k_means_after_k_centers.png", dpi=100)
    else:
        fig.savefig("balanced_k_means.png", dpi=100)
    return idxLstPerClusterDict, centerIdLst, newClusterSum

### ----- ###


#0
def selective_balanced_k_means_for_distance_2d_array(link2dArray: np.ndarray, dist2dArray: np.ndarray, nCluster: int,
        maxClusterWeight: float, weightArray=None, selectiveMethod="k_centers", isRandom=True) -> tuple:
    idxLstPerClusterDict, centerIdLst = dict(), []
    nSample = dist2dArray.shape[0]
    # print("D", nCluster)
    if nCluster > nSample:
        for iSample in range(nSample):
            idxLstPerClusterDict[iSample] = [iSample]
        centerIdLst = list(range(nSample))
        return idxLstPerClusterDict, centerIdLst
    else:
        if selectiveMethod == "k_centers":
            _, initCenterIdLst = k_centers_for_distance_2d_array(dist2dArray, nCluster, isRandom)
        else:
            initCenterIdLst = sample(list(range(nSample)), nCluster)
        # print("D", initCenterIdLst)
    if weightArray is None:
        weightArray = np.ones((nSample))
    elif weightArray.shape[0] != dist2dArray.shape[0]:
        print("Fatal error! The shape of weight array is not consistent with that of distance matrix.")
        sys.exit(1)
    idxLstPerClusterDict, centerIdLst, _ = balanced_k_means_for_distance_2d_array_with_initial_assignment(\
            link2dArray, maxClusterWeight, weightArray, initCenterIdLst)
    return idxLstPerClusterDict, centerIdLst

#1-1
def k_centers_for_distance_2d_array(dist2dArray: np.ndarray, nCluster: int, isRandom=True) -> tuple:
    nUnit = dist2dArray.shape[0]
    centerIdLst = get_initial_cluster(nUnit, isRandom)
    while len(centerIdLst) < nCluster:
        iUnit = get_idx_with_minimal_noncenter_to_center_entry(dist2dArray, centerIdLst)
        centerIdLst.append(iUnit)
    # centerIdLst.sort()
    idxLstPerClusterDict = assign_noncenter_units_to_centers(dist2dArray, centerIdLst)
    return idxLstPerClusterDict, centerIdLst

#2
def get_initial_cluster(nUnit: int, isRandom: bool) -> list:
    if isRandom:
        centerIdLst = [sample(range(nUnit), 1)[0]]
    else:
        centerIdLst = [0]
    return centerIdLst

#2
def get_idx_with_minimal_noncenter_to_center_entry(dist2dArray: np.ndarray, centerIdLst: list) -> int:
    nUnit = dist2dArray.shape[0]
    nonCenterIdLst = [i for i in range(nUnit) if i not in centerIdLst]
    minNonCenterToCenterArray = np.min(dist2dArray[nonCenterIdLst,:][:,centerIdLst], axis=1)
    # print("Dc", centerIdLst)
    # print("Dn", nonCenterIdLst)
    # print("Dm", minNonCenterToCenterArray)
    maxMinVal = np.max(minNonCenterToCenterArray)
    rowLst, colLst = np.where(dist2dArray == maxMinVal)
    for iRow, iCol in zip(rowLst, colLst):
        if iRow in centerIdLst and iCol in nonCenterIdLst:
            return iCol
        elif iRow in nonCenterIdLst and iCol in centerIdLst:
            return iRow
        else:
            print("Fatal error! Unexpected case for k-centers.")
            sys.exit(1)
    print("Fatal error! Fail to find minimal entry for k-centers.")
    sys.exit(1)

#2 TODO
def assign_noncenter_units_to_centers(dist2dArray: np.ndarray, centerIdLst: list) -> dict:
    idxLstPerClusterDict = dict()
    for centerId in centerIdLst:
        idxLstPerClusterDict[centerId] = []
    nUnit = dist2dArray.shape[0]
    for iUnit in range(nUnit):
        idx = np.argmin(dist2dArray[iUnit, centerIdLst])
        centerId = centerIdLst[idx]
        idxLstPerClusterDict[centerId].append(iUnit)
    return idxLstPerClusterDict

### --- ###
#0
def k_means_full(featureArray: np.ndarray, nCluster: int) -> tuple:
    nSample = featureArray.shape[0]
    if nSample < nCluster:
        iLst = list(range(nSample))
        d = dict()
        for i in iLst:
            d[i] = [i]
        return d, iLst
    else:
        dist2dArray = get_distance_from_array(featureArray)
        idxLstPerClusterDict, centerIdLst = \
                k_means_for_distance_2d_array(dist2dArray, nCluster)
        labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
        return labelLst, centerIdLst

#1
def k_means_for_distance_2d_array(dist2dArray: np.ndarray, nCluster: int) -> tuple:
    nSample = dist2dArray.shape[0]
    if nSample > nCluster:
        initCenterIdLst = sample(list(range(nSample)), nCluster)
        idxLstPerClusterDict, centerIdLst = \
                k_means_for_distance_2d_array_with_initial_assignment(dist2dArray, initCenterIdLst)
        return idxLstPerClusterDict, centerIdLst
    else:
        print("Warning! Too large cluster number and too small sample number.")
        return list(range(nSample)), list(range(nSample))
    

#0
def k_means_full_plot(featureArray: np.ndarray, nCluster: int) -> tuple:
    nSample = featureArray.shape[0]
    if nSample >= nCluster:
        initCenterIdLst = sample(list(range(nSample)), nCluster)
    else:
        print("Warning! Too large cluster number and too small sample number.")
        return list(range(nSample)), list(range(nSample))
    dist2dArray = get_distance_from_array(featureArray)
    idxLstPerClusterDict, centerIdLst = \
            k_means_for_distance_2d_array_with_initial_assignment_plot(featureArray, dist2dArray, initCenterIdLst)
    labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
    return labelLst, centerIdLst

#1
def k_means_for_distance_2d_array_with_initial_assignment_plot(featureArray: np.ndarray, 
        dist2dArray: np.ndarray, initCenterIdLst: list) -> tuple:
    nRow, nCol = 4, 5
    fig, axes = plt.subplots(nRow, nCol, figsize=(nCol*4, nRow*4))
    iStep = 0
    print("D1")
    idxLstPerClusterDict = get_index_list_per_cluster_dict_no_weight(dist2dArray, initCenterIdLst)
    print("D2")
    oldClusterSum = 99999999999999999.0
    newClusterSum = get_cluster_sum(dist2dArray, initCenterIdLst, idxLstPerClusterDict)
    labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, initCenterIdLst)
    plot_points_and_centers(axes[int(iStep/nCol)][iStep%nCol], featureArray, labelLst, initCenterIdLst, 
                            "{0}: {1:.4e}".format(iStep, newClusterSum))
    while abs(oldClusterSum - newClusterSum) > abs(oldClusterSum) * MINDEVIATION:
        print("D3", oldClusterSum)
        oldClusterSum = newClusterSum
        iStep += 1
        centerIdLst = get_new_center_id_list(dist2dArray, idxLstPerClusterDict)
        idxLstPerClusterDict = get_index_list_per_cluster_dict_no_weight(dist2dArray, centerIdLst)
        newClusterSum = get_cluster_sum(dist2dArray, centerIdLst, idxLstPerClusterDict)
        labelLst = get_lable_list_from_cluster_dict(idxLstPerClusterDict, centerIdLst)
        plot_points_and_centers(axes[int(iStep/nCol)][iStep%nCol], featureArray, labelLst, centerIdLst, 
                                "{0}: {1:.4e}".format(iStep, newClusterSum))
    fig.savefig("k_means.png", dpi=100)
    return idxLstPerClusterDict, centerIdLst

#1
def get_distance_from_array(featureArray: np.ndarray) -> np.ndarray:
    nSample = featureArray.shape[0]
    dist2dArray = np.zeros((nSample, nSample))
    for iFeature, uFeature in enumerate(featureArray):
        for jFeature, vFeature in enumerate(featureArray):
            dist2dArray[iFeature, jFeature] = np.sqrt(np.sum((uFeature - vFeature)**2))
    return dist2dArray

#1
def k_means_for_distance_2d_array_with_initial_assignment(dist2dArray: np.ndarray, initCenterIdLst: list) -> tuple:
    # initCenterIdLst.sort()
    idxLstPerClusterDict = get_index_list_per_cluster_dict_no_weight(dist2dArray, initCenterIdLst)
    oldClusterSum = 99999999999999999.0
    newClusterSum = get_cluster_sum(dist2dArray, initCenterIdLst, idxLstPerClusterDict)
    iCycle = 0
    while abs(oldClusterSum - newClusterSum) > abs(oldClusterSum) * MINDEVIATION:
        oldClusterSum = newClusterSum
        centerIdLst = get_new_center_id_list(dist2dArray, idxLstPerClusterDict)
        idxLstPerClusterDict = get_index_list_per_cluster_dict_no_weight(dist2dArray, centerIdLst)
        newClusterSum = get_cluster_sum(dist2dArray, centerIdLst, idxLstPerClusterDict)
        iCycle += 1
        if iCycle > MAXKMEANCYCLE:
            return idxLstPerClusterDict, centerIdLst
    return idxLstPerClusterDict, centerIdLst

#2 TODO may be faster
def get_index_list_per_cluster_dict_no_weight(dist2dArray: np.ndarray, centerIdLst: list) -> dict:
    tmp2dArray = deepcopy(dist2dArray)
    for centerId1 in centerIdLst:
        for centerId2 in centerIdLst:
            tmp2dArray[centerId1, centerId2] = CLUSTERMAXVAL
    centerRelatedMask = np.zeros(tmp2dArray.shape, dtype=bool)
    idxLstPerClusterDict = dict()
    for centerId in centerIdLst:
        idxLstPerClusterDict[centerId] = [centerId]
        centerRelatedMask[centerId, :] = True
        centerRelatedMask[:, centerId] = True
    while not np.all(tmp2dArray == CLUSTERMAXVAL):
        idx, centerId = get_idx_with_minimal_center_related_entry(tmp2dArray, centerRelatedMask, centerIdLst)
        idxLstPerClusterDict[centerId].append(idx)
        idxLstPerClusterDict[centerId].sort()
        tmp2dArray[idx, :] = CLUSTERMAXVAL
        tmp2dArray[:, idx] = CLUSTERMAXVAL
    return idxLstPerClusterDict

#1
def get_lable_list_from_cluster_dict(idxLstPerClusterDict: dict, centerIdLst: list) -> list:
    n = 0
    for centerId in centerIdLst:
        n += len(idxLstPerClusterDict[centerId])
    labelLst = [0] * n
    for iCluster, centerId in enumerate(centerIdLst):
        for idx in idxLstPerClusterDict[centerId]:
            labelLst[idx] = iCluster
    return labelLst

#1
def balanced_k_means_for_distance_2d_array_with_initial_assignment(dist2dArray: np.ndarray, 
        maxClusterWeight: float, weightArray: np.ndarray, initCenterIdLst: list) -> tuple:
    # initCenterIdLst.sort()
    idxLstPerClusterDict = get_index_list_per_cluster_dict(dist2dArray, initCenterIdLst, 
                                                           maxClusterWeight, weightArray)
    oldClusterSum = 99999999999999999.0
    newClusterSum = get_cluster_sum(dist2dArray, initCenterIdLst, idxLstPerClusterDict)
    while abs(oldClusterSum - newClusterSum) > abs(oldClusterSum) * MINDEVIATION:
        # print("D-", oldClusterSum, newClusterSum)
        # print("D+", oldClusterSum - newClusterSum, abs(oldClusterSum) * MINDEVIATION)
        oldClusterSum = newClusterSum
        centerIdLst = get_new_center_id_list(dist2dArray, idxLstPerClusterDict)
        # print("D", centerIdLst)
        idxLstPerClusterDict = get_index_list_per_cluster_dict(dist2dArray, centerIdLst, 
                                                               maxClusterWeight, weightArray)
        newClusterSum = get_cluster_sum(dist2dArray, centerIdLst, idxLstPerClusterDict)
        # print("Dn", oldClusterSum, newClusterSum)
    return idxLstPerClusterDict, centerIdLst, newClusterSum

#2
def get_index_list_per_cluster_dict(dist2dArray: np.ndarray, centerIdLst: list,
                                    maxClusterWeight: int, weightArray: np.ndarray) -> dict:
    tmp2dArray = deepcopy(dist2dArray)
    for centerId1 in centerIdLst:
        for centerId2 in centerIdLst:
            tmp2dArray[centerId1, centerId2] = CLUSTERMAXVAL
    centerRelatedMask = np.zeros(tmp2dArray.shape, dtype=bool)
    idxLstPerClusterDict = dict()
    for centerId in centerIdLst:
        idxLstPerClusterDict[centerId] = [centerId]
        centerRelatedMask[centerId, :] = True
        centerRelatedMask[:, centerId] = True
    while not np.all(tmp2dArray == CLUSTERMAXVAL):
        idx, centerId = get_idx_with_minimal_center_related_entry(tmp2dArray, centerRelatedMask, centerIdLst)
        if np.sum(weightArray[idxLstPerClusterDict[centerId] + [idx]]) <= maxClusterWeight:
            idxLstPerClusterDict[centerId].append(idx)
            idxLstPerClusterDict[centerId].sort()
            tmp2dArray[idx, :] = CLUSTERMAXVAL
            tmp2dArray[:, idx] = CLUSTERMAXVAL
        else:
            tmp2dArray[idx, centerId] = CLUSTERMAXVAL
            tmp2dArray[centerId, idx] = CLUSTERMAXVAL
    return idxLstPerClusterDict

#3
def get_idx_with_minimal_center_related_entry(dist2dArray: np.ndarray, 
                                              centerRelatedMask: np.ndarray, centerIdLst: list) -> tuple:
    centerRelatedMin = np.min(dist2dArray[centerRelatedMask])
    rowLst, colLst = np.where(dist2dArray == centerRelatedMin)
    for iRow, iCol in zip(rowLst, colLst):
        if centerRelatedMask[iRow][iCol]:
            if iRow in centerIdLst and iCol not in centerIdLst:
                return iCol, iRow
            elif iRow not in centerIdLst and iCol in centerIdLst:
                return iRow, iCol
            else:
                print("Fatal error! Unexpected case when using selective balanced k_means.")
                sys.exit(1)

#2
def get_cluster_sum(dist2dArray: np.ndarray, centerIdLst: list, idxLstPerClusterDict: dict) -> float:
    clusterSum = 0.0
    for centerId in centerIdLst:
        clusterSum += np.sum(dist2dArray[centerId, idxLstPerClusterDict[centerId]])
    return clusterSum

#2 
def get_new_center_id_list(dist2dArray: np.ndarray, idxLstPerClusterDict: dict) -> list:
    centerIdLst = []
    for centerId in idxLstPerClusterDict.keys():
        newCenterId = get_new_center_from_distance_2d_array_globally(dist2dArray, idxLstPerClusterDict[centerId])
        centerIdLst.append(newCenterId)
    return centerIdLst

#3
def get_new_center_from_distance_2d_array_globally(dist2dArray: np.ndarray, idxLst: list) -> int:
    selectedInterUnitDist2dArray = dist2dArray[idxLst,:]
    iMin = np.argmin(selectedInterUnitDist2dArray.sum(axis=0))
    return iMin

#3
def get_new_center_from_distance_2d_array_locally(dist2dArray: np.ndarray, idxLst: list) -> int:
    selectedInterUnitDist2dArray = dist2dArray[idxLst,:][:,idxLst]
    iMin = np.argmin(selectedInterUnitDist2dArray.sum(axis=1))
    return idxLst[iMin]

#3 obsolete
def get_2d_array_of_minimal_entry(dist2dArray: np.ndarray) -> list:
    idx = np.argmin(dist2dArray)
    n = dist2dArray.shape[1]
    return int(idx/n), idx%n
### --- ###


