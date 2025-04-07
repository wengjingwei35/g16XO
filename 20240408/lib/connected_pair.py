from lib.misc import flatten


def get_neighbor_layer_dict_from_connected_pair_list(connectedPairLst: list) -> list:
    neighborIdxSetLayers = dict()
    neighborIdxSetLayers[1] = get_neighbor_layer_1_dict(connectedPairLst)
    neighborIdxSetLayers[2] = get_neighbor_layer_2_dict(neighborIdxSetLayers[1])
    neighborIdxSetLayers[3] = get_neighbor_layer_3_dict(neighborIdxSetLayers[1], neighborIdxSetLayers[2])
    neighborIdxSetLayers[4] = get_neighbor_layer_4_dict(neighborIdxSetLayers[1], neighborIdxSetLayers[2], 
                                                        neighborIdxSetLayers[3])
    neighborIdxSetLayers[5] = get_neighbor_layer_5_dict(neighborIdxSetLayers[1], neighborIdxSetLayers[2], 
                                                        neighborIdxSetLayers[3], neighborIdxSetLayers[4])
    #neighborIdxSetLayer0to3s = get_zero_to_3rd_layer_neighbor_string_dict(connectedPairLst, neighborIdxSetLayer1s)
    return neighborIdxSetLayers
    #neighborIdxSetLayer0to3s
"""
def get_neighbor_layer_dict3(connectedPairLst: list) -> tuple:
    neighborIdxSetLayer1s = get_neighbor_layer_1_dict(connectedPairLst)
    neighborIdxSetLayer2s = get_neighbor_layer_2_dict(neighborIdxSetLayer1s)
    neighborIdxSetLayer3s = get_neighbor_layer_3_dict(neighborIdxSetLayer1s, neighborIdxSetLayer2s)
    neighborIdxSetLayer0to3s = get_zero_to_3rd_layer_neighbor_string_dict(connectedPairLst, neighborIdxSetLayer1s)
    return neighborIdxSetLayer1s, neighborIdxSetLayer2s, neighborIdxSetLayer3s, neighborIdxSetLayer0to3s
"""

def get_neighbor_layer_1_dict(connectedPairLst: list) -> dict:
    neighborIdxSetLayer1s = initialize_neighbor_layer_1_dict(connectedPairLst)
    for uPair in connectedPairLst:
        neighborIdxSetLayer1s[uPair[0]].add(uPair[1])
        neighborIdxSetLayer1s[uPair[1]].add(uPair[0])
    return neighborIdxSetLayer1s

def initialize_neighbor_layer_1_dict(connectedPair: list) -> dict:
    neighborIdxSetLayers = dict()
    unitLst = get_unit_list_from_pair_list(connectedPair)
    for unit in unitLst:
        neighborIdxSetLayers[unit] = set()
    return neighborIdxSetLayers

def get_unit_list_from_pair_list(connectedPair: list) -> list:
    return list(flatten(connectedPair))

def get_neighbor_layer_2_dict(neighborIdxSetLayer1s: dict) -> dict:
    neighborIdxSetLayer2s = initialize_neighbor_index_set_layer_dict_from_layer_1(neighborIdxSetLayer1s)
    for centerUnit in neighborIdxSetLayer1s.keys():
        excludeSet = {centerUnit}.union(neighborIdxSetLayer1s[centerUnit])
        for periUnit in neighborIdxSetLayer1s[centerUnit]:
            for testUnit in neighborIdxSetLayer1s[periUnit]:
                if testUnit not in excludeSet:
                    neighborIdxSetLayer2s[centerUnit].add(testUnit)
    return neighborIdxSetLayer2s

def initialize_neighbor_index_set_layer_dict_from_layer_1(neighborIdxSetLayer1s: dict) -> dict:
    neighborIdxSetLayers = dict()
    for unit in neighborIdxSetLayer1s.keys():
        neighborIdxSetLayers[unit] = set()
    return neighborIdxSetLayers

def get_neighbor_layer_3_dict(neighborIdxSetLayer1s: dict, neighborIdxSetLayer2s: dict) -> dict:
    neighborIdxSetLayer3s = initialize_neighbor_index_set_layer_dict_from_layer_1(neighborIdxSetLayer1s)
    for centerUnit in neighborIdxSetLayer1s.keys():
        excludeSet = {centerUnit}.union(neighborIdxSetLayer1s[centerUnit]).union(neighborIdxSetLayer2s[centerUnit])
        for periUnit in neighborIdxSetLayer2s[centerUnit]:
            for testUnit in neighborIdxSetLayer1s[periUnit]:
                if testUnit not in excludeSet:
                    neighborIdxSetLayer3s[centerUnit].add(testUnit)
    return neighborIdxSetLayer3s

def get_neighbor_layer_4_dict(neighborIdxSetLayer1s: dict, neighborIdxSetLayer2s: dict, neighborIdxSetLayer3s: dict)\
        -> dict:
    neighborIdxSetLayer4s = initialize_neighbor_index_set_layer_dict_from_layer_1(neighborIdxSetLayer1s)
    for centerUnit in neighborIdxSetLayer1s.keys():
        excludeSet = {centerUnit}.union(neighborIdxSetLayer1s[centerUnit]).union(neighborIdxSetLayer2s[centerUnit])\
                     .union(neighborIdxSetLayer3s[centerUnit])
        for periUnit in neighborIdxSetLayer3s[centerUnit]:
            for testUnit in neighborIdxSetLayer1s[periUnit]:
                if testUnit not in excludeSet:
                    neighborIdxSetLayer4s[centerUnit].add(testUnit)
    return neighborIdxSetLayer4s

def get_neighbor_layer_5_dict(neighborIdxSetLayer1s: dict, neighborIdxSetLayer2s: dict, neighborIdxSetLayer3s: dict,\
        neighborIdxSetLayer4s: dict) -> dict:
    neighborIdxSetLayer5s = initialize_neighbor_index_set_layer_dict_from_layer_1(neighborIdxSetLayer1s)
    for centerUnit in neighborIdxSetLayer1s.keys():
        excludeSet = {centerUnit}.union(neighborIdxSetLayer1s[centerUnit]).union(neighborIdxSetLayer2s[centerUnit])\
                     .union(neighborIdxSetLayer3s[centerUnit]).union(neighborIdxSetLayer4s[centerUnit])
        for periUnit in neighborIdxSetLayer4s[centerUnit]:
            for testUnit in neighborIdxSetLayer1s[periUnit]:
                if testUnit not in excludeSet:
                    neighborIdxSetLayer5s[centerUnit].add(testUnit)
    return neighborIdxSetLayer5s

def get_zero_to_3rd_layer_neighbor_string_dict(connectedPairLst, neighborIdxSetLayer1s) -> dict:
    neighborIdxSetLayer0to3s = initialize_neighbor_index_set_layer_dict_from_layer_1(neighborIdxSetLayer1s)
    return neighborIdxSetLayer0to3s

