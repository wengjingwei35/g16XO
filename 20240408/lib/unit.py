import sys
import numpy as np
from copy import deepcopy
from file.gaussian import read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file
from file.gaussian import ensure_consistency_between_unit_file_and_gjf_file
from file.other import print_xo_log, read_connectivity_file, is_valid_connectivity_file
from file.unit import read_unit_file, write_unit_file
from lib.misc import get_number_index_list_from_serial_strings, print_error_and_exit
from lib.misc import get_number_of_heavy_atoms_from_element_list
from lib.misc import change_dict_key
from lib.misc import get_center_unit_index_from_egg_name
from lib.connected_pair import get_neighbor_layer_dict_from_connected_pair_list
from file.unit import renew_additional_unit_file_and_return_index_of_new_unit


class Unit():
    def __init__(self, unitFile: str, neighborType="C", connectivityCutoff=0.002, connectivityFile="connectivity"):
        self.UnitFile = unitFile
        self.Type = neighborType
        self.ConnectivityCutoff = connectivityCutoff
        self.ConnectivityFile = connectivityFile
        self.TotAtom, self.AtomIdxUnitLst, self.UnitIdxLst, self.UnitChargeLst, self.UnitSpinLst \
                = read_unit_file(self.UnitFile)
        self.N = len(self.UnitIdxLst)
        self.NUnit = len(self.UnitIdxLst)
        self.AtomIdx2UnitIdx = self.get_atom_index_to_unit_index_dict()
        if is_valid_connectivity_file(self.ConnectivityFile):
            self.get_neighbor() #UnitLayerSets, unitNeighborhoodSets, unitNeighborhoodLsts[iLayer][centerUnitId]
            self.UnitNeighborIdxSetLstsAsLevelLst = self.get_unit_neighborhood_set_list_dict_as_level_list()
            self.MaxEggLevel = len(self.UnitNeighborIdxSetLstsAsLevelLst) - 1
            self.NumEggLevel = len(self.UnitNeighborIdxSetLstsAsLevelLst)
            
    ##### ----- #####
    #0
    def get_radius_vs_size_plots(self):
        import matplotlib.pyplot as plt
        self.PlotRadiusLst = np.linspace(1.5, 6, 10)
        self.get_radius_vs_fragment_size_plot()
        self.get_radius_vs_subsystem_size_plot()
    
    #1
    def get_radius_vs_fragment_size_plot(self):
        pass
    
    #1
    def get_radius_vs_subsystem_size_plot(self):
        pass
    
    ##### ----- #####
    
    #0
    def get_unit_neighborhood_set_list_dict_as_level_list(self) -> list:
        # if self.Type == "CS":
        #     return [self.UnitNeighborhoodSets[0], self.UnitNeighborhoodSets[1], \
        #             self.UnitNeighborhoodSets[2], self.UnitNeighborhoodSets[3]]
        if self.Type == "C":
            return [self.UnitNeighborhoodSets[1], self.UnitNeighborhoodSets[2], self.UnitNeighborhoodSets[3]]
        else:
            print(" Fatal error! Invalid unit connectivity type =", self.Type)
            sys.exit(1)

    def get_number_of_atom_arrays_of_unit_index_set_list_from_gjf_file(self, gjfFile: str, 
                                                                       unitIdxSetLst: list) -> tuple:
        ensure_consistency_between_unit_file_and_gjf_file(self.UnitFile, gjfFile)
        _, _, _, elementLst, _ = read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(gjfFile)
        nHeavyAtomLst, nTotAtomLst = [], []
        for unitIdxSet in unitIdxSetLst:
            selElementLst = []
            for iUnit in unitIdxSet:
                selElementLst.extend([elementLst[iAtom] for iAtom in self.AtomIdxUnitLst[iUnit]])
            nHeavyAtomLst.append(get_number_of_heavy_atoms_from_element_list(selElementLst))
            nTotAtomLst.append(len(selElementLst))
        return np.array(nHeavyAtomLst), np.array(nTotAtomLst)

    ###
    def get_atom_index_to_unit_index_dict(self) -> dict:
        atomIdx2UnitIdx = dict()
        for iUnit, atomIdxLstInUnit in enumerate(self.AtomIdxUnitLst):
            for iAtom in atomIdxLstInUnit:
                atomIdx2UnitIdx[iAtom] = iUnit
        return atomIdx2UnitIdx
    ##### ----- #####

    ##### ----- #####
    #0
    def get_neighbor(self):
        self.get_connected_pair_list_from_connectivityfile(self.UnitIdxLst) #connectedPairLst
        self.UnitLayerSets = get_neighbor_layer_dict_from_connected_pair_list(self.ConnectedPairLst)
        self.get_unit_neighborhood_set_dict() #unitNeighborhoodSets[iLayer][centerUnitId]
        self.get_unit_neighborhood_list_dict() #unitNeighborhoodLsts[iLayer][centerUnitId]
    
    #1
    def get_connected_pair_list_from_connectivityfile(self, unitIdxLst: list):
        connectivityPair, connectivity = read_connectivity_file(self.ConnectivityFile)
        self.ConnectedPairLst = self.get_connected_pair_list_from_connectivity_and_pair_list(\
                connectivityPair, connectivity, self.ConnectivityCutoff)
        self.ensure_each_unit_connected_with_one_unit(self.ConnectedPairLst, unitIdxLst)
    
    #2
    def get_connected_pair_list_from_connectivity_and_pair_list(self, connectivityPairLst: list, 
                                                                connectivityLst: list, 
                                                                connectivityCutoff: float) -> list:
        connectedPairLst = []
        for uPair, uConnectivity in zip(connectivityPairLst, connectivityLst):
            if uConnectivity >= connectivityCutoff:
                connectedPairLst.append(uPair)
        return connectedPairLst

    #2
    def ensure_each_unit_connected_with_one_unit(self, connectedPairLst: list, unitIdxLst: list) -> None:
        remainedUnitIdxLst = deepcopy(unitIdxLst)
        for u, v in connectedPairLst:
            if u in remainedUnitIdxLst:
                remainedUnitIdxLst.remove(u)
            if v in remainedUnitIdxLst:
                remainedUnitIdxLst.remove(v)
        if len(remainedUnitIdxLst) > 0:
            print_error_and_exit("Invalid connectivity file =", self.ConnectivityFile, 
                    "with no unit indexed with", ",".join([str(i) for i in remainedUnitIdxLst]) + ".", 
                    "Please remove the file and use larger distance cutoff for unit pair (-d) or",
                    "smaller connectivity cutoff (-cc).")
    
    #1
    def get_unit_neighborhood_set_dict(self): #-> self.UnitNeighborhoodSets[iLayer][centerUnitId]
        MAXLAYER = 4
        self.UnitNeighborhoodSets = dict()
        for iLayer in range(1, MAXLAYER+1):
            self.UnitNeighborhoodSets[iLayer] = dict()
        for centerUnitId in self.UnitLayerSets[1].keys():
            self.UnitNeighborhoodSets[1][centerUnitId] = [{centerUnitId} | self.UnitLayerSets[1][centerUnitId]]
            for iLayer in range(2, MAXLAYER+1):
                self.UnitNeighborhoodSets[iLayer][centerUnitId] = \
                        [self.UnitNeighborhoodSets[iLayer-1][centerUnitId][0] \
                        | self.UnitLayerSets[iLayer][centerUnitId]]
    
    #1
    def get_unit_neighborhood_list_dict(self): #-> self.unitNeighborhoodLsts[iLayer][[centerUnitId]]
        self.UnitNeighborhoodLsts = dict()
        for uLayer in self.UnitNeighborhoodSets.keys():
            self.UnitNeighborhoodLsts[uLayer] = dict()
            for uid in self.UnitNeighborhoodSets[uLayer]:
                self.UnitNeighborhoodLsts[uLayer][uid] = [sorted(self.UnitNeighborhoodSets[uLayer][uid])]
    ##### ----- #####

    ##### ----- #####
    def get_unit_index_list_from_atom_index_list(self, atomIdxLst: list) -> list:
        unitIdxSet = set()
        for iAtom in atomIdxLst:
            unitIdxSet.add(self.get_unit_index_from_atom_index(iAtom))
        return sorted(unitIdxSet)
    
    def get_unit_index_from_atom_index(self, atomIdx: int) -> int:
        for iUnit, atomIdxLst in zip(self.UnitIdxLst, self.AtomIdxUnitLst):
            if atomIdx in atomIdxLst:
                return iUnit
        print("Fatal error! No atom index is found in current unitfile =", self.UnitFIle)
        sys.exit(0)
    ##### ----- #####

    ##### ----- #####
    def get_atom_index_list_from_unit_index_list(self, unitIdxLst: list) -> list:
        atomIdxLst = []
        for iUnit in sorted(unitIdxLst):
            atomIdxLst.extend(self.AtomIdxUnitLst[iUnit])
        return atomIdxLst

    def get_atom_index_set_from_unit_index_set(self, unitIdxSet: set) -> list:
        return set(self.get_atom_index_list_from_unit_index_list(unitIdxSet))

    def get_atom_index_list_of_lists_from_unit_index_list_of_lists(self, unitIdxSetInLst: list) -> list:
        atomIdxLstInLst = []
        for unitIdxSet in unitIdxSetInLst:
            atomIdxLst = sorted(self.get_atom_index_list_from_unit_index_list(unitIdxSet))
            atomIdxLstInLst.append(atomIdxLst)
        return atomIdxLstInLst
    ##### ----- #####

    ##### ----- #####
    #0
    def get_atom_serial_list_of_lists_from_unit_index_list_of_lists(self, unitIdxSetInLst: list) -> list:
        atomSerialLstInLst = []
        for unitIdxSet in unitIdxSetInLst:
            atomSerialLst = sorted(self.get_atom_serial_list_from_unit_index_list(unitIdxSet))
            atomSerialLstInLst.append(atomSerialLst)
        return atomSerialLstInLst
    
    #1
    def get_atom_serial_list_from_unit_index_list(self, unitIdxLst: list) -> list:
        atomIdxLst = self.get_atom_index_list_from_unit_index_list(unitIdxLst)
        return [i + 1 for i in atomIdxLst]
    ##### ----- #####

    ##### ----- #####
    #####
    def get_weight_index_and_name_lists_of_derived_fragments(self, eggNameLst: list, \
                                                                   eggIdxSetInLst: list) -> tuple:
        print("------------------------ Fragment generation ------------------------")
        fragmentWeightLst = [1] * len(eggIdxSetInLst)
        fragmentUnitIdxSetLst = deepcopy(eggIdxSetInLst)
        fragmentNameLst = deepcopy(eggNameLst)
        currentLayerUnitIdxSetLst = deepcopy(eggIdxSetInLst)
        currentLayerNameLst = deepcopy(eggNameLst)
        currentLayerMaxEggIdxLst = list(range(len(eggNameLst)))
        currentWeight = -1
        uniqueUnitIdxSetLst = deepcopy(eggIdxSetInLst)
        isAllVoidSet = False
        while not isAllVoidSet:
            isAllVoidSet = True
            nextLayerUnitIdxSetLst, nextLayerNameLst, nextLayerMaxEggIdxLst = [], [], []
            self.print_fragment_layer_statistics(currentLayerNameLst, uniqueUnitIdxSetLst)
            for currentName, currentUnitIdxSet, currentMaxEggIdx in zip(currentLayerNameLst, \
                                                                        currentLayerUnitIdxSetLst, \
                                                                        currentLayerMaxEggIdxLst):
                for iEgg, (eggName, eggIdxSet) in enumerate(zip(eggNameLst, eggIdxSetInLst)):
                    if iEgg > currentMaxEggIdx:
                        newIdxSet = currentUnitIdxSet & eggIdxSet
                        if newIdxSet != set():
                            newName = self.append_new_name_to_fragment_name_string(currentName, eggName)
                            fragmentWeightLst.append(currentWeight)
                            fragmentUnitIdxSetLst.append(newIdxSet)
                            fragmentNameLst.append(newName)
                            nextLayerUnitIdxSetLst.append(newIdxSet)
                            nextLayerNameLst.append(newName)
                            nextLayerMaxEggIdxLst.append(iEgg)
                            isAllVoidSet = False
                            if newIdxSet not in uniqueUnitIdxSetLst:
                                uniqueUnitIdxSetLst.append(newIdxSet)
            currentWeight *= -1
            currentLayerNameLst = nextLayerNameLst
            currentLayerUnitIdxSetLst = nextLayerUnitIdxSetLst
            currentLayerMaxEggIdxLst = nextLayerMaxEggIdxLst
        print("---------------------------------------------------------------------")
        return fragmentWeightLst, fragmentUnitIdxSetLst, fragmentNameLst
    
    ###
    def print_fragment_layer_statistics(self, currentLayerNameLst: list, uniqueUnitIdxSetLst: list):
        iLayer =currentLayerNameLst[0].count(".") + 1
        print(" Layer {0:2d} has {1:5d} fragments, {2:5d} unique ones in total".format(iLayer, 
                                                                                       len(currentLayerNameLst),
                                                                                       len(uniqueUnitIdxSetLst)))
    
    ###
    def append_new_name_to_fragment_name_string(self, nameStr: str, vName: str) -> str:
        nameLst = nameStr.split(".") + [vName]
        return ".".join(sorted(nameLst))
    ##### ----- #####

    ##### ----- #####
    #0
    def write(self, unitFile: str):
        write_unit_file(unitFile, self.TotAtom, self.AtomIdxUnitLst, self.UnitIdxLst, \
                        self.UnitChargeLst, self.UnitSpinLst)
    ##### ----- #####


class DistUnit(Unit):
    def __init__(self, unitFile: str, gjfFile: str, initRadius=3.0, incrRadius=2.0, deltaRadius=0.1, \
                 maxIncrHvyAtom=100, nEggLevel=3, rpaLst=[], radiusMode="r", maxAtom=99999999):
        print(" Generating distance unit ...")
        self.UnitFile = unitFile
        self.GjfFile = gjfFile
        self.InitRadius = initRadius
        self.IncrRadius = incrRadius
        self.DeltaRadius = deltaRadius
        self.MaxIncrHvyAtom = maxIncrHvyAtom
        self.NEggLevel = nEggLevel
        self.RadiusOrNPercentOrNAtomLst = rpaLst
        self.RadiusMode = radiusMode
        self.MaxAtom = maxAtom
        self.get_radius_or_npercent_or_natom_list()
        # self.MaxEggLevel = nEggLevel - 1
        self.Charge, self.Spin, self.Xyz, self.ElementLst, self.BondOrders = \
                read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(self.GjfFile)
        self.TotAtom, self.AtomIdxUnitLst, self.UnitIdxLst, self.UnitChargeLst, self.UnitSpinLst \
                = read_unit_file(self.UnitFile)
        self.N = len(self.UnitIdxLst)
        self.MinDist2dArray = self.get_minimal_distance_2d_array_between_two_units()
        self.AtomIdx2UnitIdx = self.get_atom_index_to_unit_index_dict()
        if self.RadiusMode == "r":
            self.UnitNeighborIdxSetLstsAsLevelLst, self.UnitNeighborRadiusDictAsLevelLst \
                    = self.get_unit_neighbor_index_sets_in_dict_as_level_list()
        print(" Generation finished.")
        
    ##### ----- #####
    #0
    def get_atom_number_from_unit_index_list(self, unitIdxLst: list) -> int:
        nAtom = 0
        for unitIdx in unitIdxLst:
            nAtom += len(self.AtomIdxUnitLst[unitIdx])
        return nAtom
    ##### ----- #####
    
    #0
    def get_egg_index_set_dict_for_each_npercent(self, nPercentLst: list) -> dict:
        nAtomLst = [round(nPrecent * 0.01 * self.TotAtom) for nPrecent in nPercentLst]
        eggIdxSetsPerNAtom, radiusDictPerNAtom = self.get_egg_index_set_dict_for_each_natom(nAtomLst)
        eggIdxSetsPerNPercent = change_dict_key(eggIdxSetsPerNAtom, nAtomLst, nPercentLst)
        radiusDictPerNPercent = change_dict_key(radiusDictPerNAtom, nAtomLst, nPercentLst)
        return eggIdxSetsPerNPercent, radiusDictPerNPercent
    
    #1 TODO is this code correct?
    def get_egg_index_set_dict_for_each_natom(self, nAtomLst: list) -> dict:
        eggIdxSetsPerNAtom, radiusDictPerNAtom = dict(), dict()
        self.NeighborUnitIdxLstPerUnit = self.get_neighbor_unit_index_list_from_near_to_far_per_unit()
        for nAtom in nAtomLst:
            eggIdxSetsPerNAtom[nAtom] = dict()
            radiusDictPerNAtom[nAtom] = dict()
            for iUnit, unitIdxLst in enumerate(self.NeighborUnitIdxLstPerUnit):
                nAtomCumArray = np.cumsum([len(self.AtomIdxUnitLst[uIdx]) for uIdx in unitIdxLst])
                maxIdx = np.max(np.where(nAtomCumArray <= nAtom))
                eggIdxSetsPerNAtom[nAtom][iUnit] = set(unitIdxLst[0:maxIdx+1])
                radiusDictPerNAtom[nAtom][iUnit] = self.MinDist2dArray[iUnit][maxIdx]
        return eggIdxSetsPerNAtom, radiusDictPerNAtom
    
    #2
    def get_neighbor_unit_index_list_from_near_to_far_per_unit(self) -> dict:
        #neighborUnitIdxLstPerUnit[iUnit] = np.argsort(minDistArray)
        return np.argsort(self.MinDist2dArray)
    
    """
    neighborAtomIdxLstPerUnit = self.get_neighbor_atom_index_list_from_near_to_far_per_unit()
    for unit in neighborAtomIdxLstPerUnit.keys():
        neighborAtomIdxLstPerUnit[unit] = \
                self.atom_index_list_to_unique_unit_index_list(neighborAtomIdxLstPerUnit[unit])
    #3
    def atom_index_list_to_unique_unit_index_list(self, atomIdxLst: list) -> list:
        uniqueUnitIdxLst = []
        for atomIdx in atomIdxLst:
            unitIdx = self.AtomIdx2UnitIdx[atomIdx]
            if unitIdx not in uniqueUnitIdxLst:
                uniqueUnitIdxLst.append(unitIdx)
        return uniqueUnitIdxLst
    """
    
    ##0
    def get_radius_or_npercent_or_natom_list(self):
        if self.RadiusMode == "r":
            self.RadiusLst = self.RadiusOrNPercentOrNAtomLst
        elif self.RadiusMode == "p":
            self.NPercentLst = self.RadiusOrNPercentOrNAtomLst
        elif self.RadiusMode == "a":
            self.NAtomLst = self.RadiusOrNPercentOrNAtomLst
        else:
            print(" Fatal error! Unknown radius mode =", self.RadiusMode)
    
    #0
    def get_unit_neighbor_index_sets_in_dict_as_level_list(self) -> tuple:
        if len(self.RadiusLst) == 0:
            return self.get_unit_neighbor_index_sets_in_dict_as_level_list_from_radius_variables()
        else:
            return self.get_unit_neighbor_index_sets_in_dict_as_level_list_from_radius_list()
    
    #1
    def get_unit_neighbor_index_sets_in_dict_as_level_list_from_radius_list(self) -> tuple:
        nEggLevel = len(self.RadiusLst)
        unitNeighborIdxSetLstsAsLevelLst = [dict() for _ in range(nEggLevel)]
        neighborRadiusDictAsLevelLst = [dict() for _ in range(nEggLevel)]
        for iUnit in self.UnitIdxLst:
            for iEggLevel, currRadius in enumerate(self.RadiusLst):
                unitNeighborIdxSet, limitedRadius = \
                        self.get_neighbor_index_set_for_a_unit_in_a_distance_with_limited_size(\
                        self.MinDist2dArray[iUnit], currRadius)
                unitNeighborIdxSetLstsAsLevelLst[iEggLevel][iUnit] = [unitNeighborIdxSet]
                neighborRadiusDictAsLevelLst[iEggLevel][iUnit] = limitedRadius
        return unitNeighborIdxSetLstsAsLevelLst, neighborRadiusDictAsLevelLst
    
    #1-1
    def get_egg_index_set_dict_for_each_radius(self) -> dict:
        unitNeighborIdxSetLstsAsLevelLst, _ = \
                self.get_unit_neighbor_index_sets_in_dict_as_level_list_from_radius_list()
        unitNeighborIdxSetsPerLevel = dict()
        for iEggLevel, unitNeighborIdxSetInLsts in enumerate(unitNeighborIdxSetLstsAsLevelLst):
            unitNeighborIdxSetsPerLevel[iEggLevel] = dict()
            for iUnit in unitNeighborIdxSetInLsts.keys():
                unitNeighborIdxSetsPerLevel[iEggLevel][iUnit] = unitNeighborIdxSetInLsts[iUnit][0]
        eggIdxSetsPerRad = change_dict_key(unitNeighborIdxSetsPerLevel, range(len(self.RadiusLst)), self.RadiusLst)
        return eggIdxSetsPerRad
    
    #1
    def get_unit_neighbor_index_sets_in_dict_as_level_list_from_radius_variables(self) -> tuple:
        unitNeighborIdxSetLstsAsLevelLst = [dict() for _ in range(self.NEggLevel)]
        nHvyAtomNeighborDictPerLevel = [dict() for _ in range(self.NEggLevel)]
        neighborRadiusDictAsLevelLst = [dict() for _ in range(self.NEggLevel)]
        nHvyAtomUnitArray = self.get_heavy_atom_number_per_unit_array()
        for iUnit in self.UnitIdxLst:
            # print(" D - generating neighbors for unit", iUnit)
            for iEggLevel in range(self.NEggLevel):
                currRadius = self.InitRadius + iEggLevel * self.IncrRadius
                unitNeighborIdxSet, limitedRadius = \
                        self.get_neighbor_index_set_for_a_unit_in_a_distance_with_limited_size(\
                        self.MinDist2dArray[iUnit], currRadius)
                nHvyAtomNeighbor = self.get_number_of_heavy_atoms_of_unit_index_set(\
                        nHvyAtomUnitArray, unitNeighborIdxSet)
                # print(" Di", iUnit, iEggLevel, currRadius, nHvyAtomNeighbor, unitNeighborIdxSet)
                if iEggLevel > 0:
                    currMaxHvyAtom = nHvyAtomNeighborDictPerLevel[0][iUnit] + iEggLevel * self.MaxIncrHvyAtom
                    # currMinRadius = self.InitRadius + (iEggLevel-1) * self.IncrRadius + self.DeltaRadius
                    currMinRadius = neighborRadiusDictAsLevelLst[iEggLevel-1][iUnit] + self.DeltaRadius * 1.001
                    # print(" Dm", iUnit, iEggLevel, currMinRadius, currMaxHvyAtom)
                    while nHvyAtomNeighbor > currMaxHvyAtom and currRadius > currMinRadius:
                        currRadius -= self.DeltaRadius
                        unitNeighborIdxSet, limitedRadius = \
                                self.get_neighbor_index_set_for_a_unit_in_a_distance_with_limited_size(\
                                self.MinDist2dArray[iUnit], currRadius)
                        nHvyAtomNeighbor = self.get_number_of_heavy_atoms_of_unit_index_set(\
                                nHvyAtomUnitArray, unitNeighborIdxSet)
                unitNeighborIdxSetLstsAsLevelLst[iEggLevel][iUnit] = [unitNeighborIdxSet]
                nHvyAtomNeighborDictPerLevel[iEggLevel][iUnit] = nHvyAtomNeighbor
                neighborRadiusDictAsLevelLst[iEggLevel][iUnit] = currRadius
                # print(" Df", iUnit, iEggLevel, currRadius, nHvyAtomNeighbor, unitNeighborIdxSet)
        # print(" Dr", neighborRadiusDictAsLevelLst[0])
        return unitNeighborIdxSetLstsAsLevelLst, neighborRadiusDictAsLevelLst
    
    ####
    def get_minimal_distance_2d_array_between_two_units(self) -> np.ndarray:
        minDists = self.get_minimal_distance_dict_between_two_units()
        minDistLstInLst = []
        for iUnit in self.UnitIdxLst:
            minDistLst = []
            for jUnit in self.UnitIdxLst:
                if iUnit < jUnit:
                    minDistLst.append(minDists[iUnit][jUnit])
                elif iUnit == jUnit:
                    minDistLst.append(0.0)
                else:
                    minDistLst.append(minDists[jUnit][iUnit])
            minDistLstInLst.append(minDistLst)
        return np.array(minDistLstInLst)
    
    ###
    def get_minimal_distance_dict_between_two_units(self) -> dict:
        minDists = dict()
        for iUnit in self.UnitIdxLst:
            minDists[iUnit] = dict()
            for jUnit in self.UnitIdxLst:
                if iUnit < jUnit:
                    minDists[iUnit][jUnit] = self.calculate_minimal_inter_atom_distance_between_two_units(\
                            self.AtomIdxUnitLst[iUnit], self.AtomIdxUnitLst[jUnit])
        return minDists
    
    ##
    def calculate_minimal_inter_atom_distance_between_two_units(self, atomLst1: list, atomLst2: list) -> float:
        minDist = float("inf")
        for i in atomLst1:
            for j in atomLst2:
                dist = self.calculate_array_distance(self.Xyz[i], self.Xyz[j])
                if dist < minDist:
                    minDist = dist
        return minDist
    
    #
    def calculate_array_distance(self, array1: np.ndarray, array2: np.ndarray) -> float:
        return np.sqrt(np.sum((array1 - array2)**2))
    
    ####
    def get_heavy_atom_number_per_unit_array(self) -> np.ndarray:
        nHvyAtomUnitLst = []
        for iUnit in range(self.N):
            selElementLst = [self.ElementLst[iAtom] for iAtom in self.AtomIdxUnitLst[iUnit]]
            nHvyAtomUnitLst.append(get_number_of_heavy_atoms_from_element_list(selElementLst))
        return np.array(nHvyAtomUnitLst)
    
    ####
    def get_neighbor_index_set_for_a_unit_in_a_distance_with_limited_size(self, 
            minDistArray: np.ndarray, dist: float) -> tuple:
        unitIdxWithinRadiusArray = np.where(minDistArray <= dist)[0]
        sortedIdxLst = np.argsort(minDistArray[unitIdxWithinRadiusArray])
        sortedUnitIdxLst = unitIdxWithinRadiusArray[sortedIdxLst]
        selUnitIdxLst = []
        nAtom = 0
        for unitIdx in sortedUnitIdxLst:
            nAtom += len(self.AtomIdxUnitLst[unitIdx])
            if nAtom > self.MaxAtom:
                if len(selUnitIdxLst) == 0:
                    print("Warning!!! One unit itself contains too many atoms \
                          whose number is beyond the selected maxAtom =", self.MaxAtom)
                # print(nAtom - len(self.AtomIdxUnitLst[unitIdx]), end=", ")
                return set(selUnitIdxLst), minDistArray[unitIdx] - 0.01
            else:
                selUnitIdxLst.append(unitIdx)
        # print(nAtom, end=", ")
        return set(selUnitIdxLst), dist
    
    ####
    def get_number_of_heavy_atoms_of_unit_index_set(self, nHvyAtomUnitArray: np.ndarray, unitIdxSet: set) -> int:
        return np.sum(nHvyAtomUnitArray[list(unitIdxSet)])
    
    ###
    def get_neighbor_index_set_for_a_unit_in_a_distance_old(self, iUnit: int, dist: float) -> dict:
        unitNeighborIdxSet = {iUnit}
        for iUnit in self.UnitIdxLst:
            for jUnit in self.UnitIdxLst:
                if jUnit != iUnit:
                    if self.if_minimal_inter_atom_distance_below_d_old(self.AtomIdxUnitLst[iUnit], 
                                                                   self.AtomIdxUnitLst[jUnit], dist):
                        unitNeighborIdxSet.add(jUnit)
        return unitNeighborIdxSet
    
    ##
    def if_minimal_inter_atom_distance_below_d_old(self, atomLst1: list, atomLst2: list, d: float) -> bool:
        for i in atomLst1:
            for j in atomLst2:
                if self.calculate_array_distance(self.Xyz[i], self.Xyz[j]) <= d:
                    return True
        return False


class RadiusUnit(DistUnit):
    def __init__(self, gjfFile: str, unitFile: str, radius: float, isLog=True):
        if isLog:
            print_xo_log(" Generating radius units at radius = {0} ...".format(radius))
        # input properties
        self.GjfFile = gjfFile
        self.UnitFile = unitFile
        self.Radius = radius
        # derived properties
        self.U = Unit(self.UnitFile)
        self.Charge, self.Spin, self.Xyz, self.ElementLst, self.BondOrders = \
                read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(self.GjfFile)
        self.TotAtom, self.AtomIdxUnitLst, self.UnitIdxLst, self.UnitChargeLst, self.UnitSpinLst = \
                read_unit_file(self.UnitFile)
        self.N = len(self.UnitIdxLst)
        self.MinDist2dArray = self.get_minimal_distance_2d_array_between_two_units()
        self.AtomIdx2UnitIdx = self.get_atom_index_to_unit_index_dict()
        self.ConnectedPairLst = self.get_connected_unit_pair_list(radius)
        self.InitialRawEggUnitIdxSets = self.define_raw_eggs_with_selected_radius(radius) # "U" + str(uid+1) + "F" + str(iFragment)
        self.StuffedRawEggUnitIdxSets = self.get_raw_stuffed_unit_eggs()
        if isLog:
            print_xo_log(" Generation finished.")
    
    #0
    def write_merged_unit_pairs_to_file(self, additionalUnitFile: str):
        for iUnit, jUnit in self.ConnectedPairLst:
            _ = renew_additional_unit_file_and_return_index_of_new_unit(\
                    additionalUnitFile, self.UnitFile, [iUnit, jUnit])
    
    #0
    def get_connected_unit_pair_list(self, radius: float) -> list:
        connectedPairLst = []
        for uIdx, distArray in enumerate(self.MinDist2dArray):
            for vIdx in np.where(distArray <= radius)[0]:
                if uIdx < vIdx:
                    connectedPairLst.append([uIdx, vIdx])
        return connectedPairLst
        
    #-0
    def define_raw_eggs_with_selected_radius(self, radius: float) -> dict:
        initUnitEggIdxSets = dict()
        iFragment = 0
        for uIdx, distArray in enumerate(self.MinDist2dArray):
            eggName = "U" + str(uIdx + 1) + "F" + str(iFragment + 1)
            initUnitEggIdxSets[eggName] = set(np.where(distArray <= radius)[0])
        return initUnitEggIdxSets
    
    #-0
    def get_raw_stuffed_unit_eggs(self) -> dict:
        maxHvAtm, maxAtom = self.get_max_atom_number_among_eggs(self.InitialRawEggUnitIdxSets)
        print_xo_log(" - max number of atoms =", maxAtom)
        print_xo_log(" - max number of heavy atoms =", maxHvAtm)
        stuffedUnitEggs = self.get_unit_egg_index_set_dict_from_max_atom(maxAtom)
        return stuffedUnitEggs
    
    #-1
    def get_max_atom_number_among_eggs(self, eggUnitIdxSets: dict) -> int:
        nHeavyAtomArray, nTotAtomArray = \
                self.U.get_number_of_atom_arrays_of_unit_index_set_list_from_gjf_file(self.GjfFile, eggUnitIdxSets.values())
        return np.max(nHeavyAtomArray), np.max(nTotAtomArray)
    
    #-1
    def get_unit_egg_index_set_dict_from_max_atom(self, maxAtom: int) -> dict:
        unitEggIdxSets = dict()
        for eggName in self.InitialRawEggUnitIdxSets.keys():
            unitIdx = get_center_unit_index_from_egg_name(eggName)
            unitEggIdxSets[eggName] = self.get_unit_egg_index_set_from_max_atom(unitIdx, maxAtom)
        return unitEggIdxSets
    
    #-2
    def get_unit_egg_index_set_from_max_atom(self, unitIdx: int, maxAtom: int) -> set:
        unitIdxArrayFromNearToFar = np.argsort(self.MinDist2dArray[unitIdx])
        nAtomCumArray = np.cumsum([len(self.AtomIdxUnitLst[uIdx]) for uIdx in unitIdxArrayFromNearToFar])
        unitEggIdxSet = set(unitIdxArrayFromNearToFar[nAtomCumArray <= maxAtom])
        if len(unitEggIdxSet) == 0:
            unitEggIdxSet = set(unitIdx)
        return unitEggIdxSet
    
    #0
    def print_initial_egg_unit_index_set_dict(self):
        print("self.InitialRawEggUnitIdxSets =")
        self.print_dict(self.InitialRawEggUnitIdxSets)

    #1
    def print_dict(self, inDict: dict):
        for key, val in inDict.items():
            print("- Key:", key)
            print("- Val:", ", ".join([str(i) for i in sorted(list(val))]))

    #0
    def print_stuffed_egg_unit_index_set_dict(self):
        print("self.StuffedRawEggUnitIdxSets =")
        self.print_dict(self.StuffedRawEggUnitIdxSets)

    #0
    def print_min_dist_2d_array(self):
        print("self.MinDist2dArray =")
        self.print_2d_array(self.MinDist2dArray)

    #1
    def print_2d_array(self, ndarray2d: np.ndarray):
        for uarray in ndarray2d:
            print("-", ", ".join(["{:.1f}".format(val) for val in uarray]))
    
    #0
    def get_egg_index_set_dict_after_merge(self, unitIdxLst: list, newUnitIdx: int) -> dict:
        eggIdxSets = dict()
        iFragment = 0
        mergedEggName = "U" + str(newUnitIdx + 1) + "F" + str(iFragment + 1)
        eggIdxSets[mergedEggName] = set([])
        unitIdxSet = set(unitIdxLst)
        for uIdx, distArray in enumerate(self.MinDist2dArray):
            if uIdx not in unitIdxLst:
                eggName = "U" + str(uIdx + 1) + "F" + str(iFragment + 1)
                eggIdxSets[eggName] = set(np.where(distArray <= self.Radius)[0])
            else:
                eggIdxSets[mergedEggName] = \
                        eggIdxSets[mergedEggName] | set(np.where(distArray <= self.Radius)[0])
        for eggName in eggIdxSets.keys():
            if len(eggIdxSets[eggName] & unitIdxSet) > 0:
                eggIdxSets[eggName] = eggIdxSets[eggName] | unitIdxSet
        return eggIdxSets
    
        
