import sys
import numpy as np
from math import ceil
from copy import deepcopy
from file.unit import write_unit_file
from lib.cluster import get_cluster_sum
from lib.unit import DistUnit, RadiusUnit
from lib.misc import remove_subset_in_dict, get_all_index_set_from_dict
from lib.misc import pair_lists_to_set_dict, dict_to_lists_in_list
from lib.cluster import selective_balanced_k_means_for_distance_2d_array
from lib.cluster import k_means_for_distance_2d_array, k_centers_for_distance_2d_array
from file.xo import write_egg_file_from_egg_dicts

LARGEINT = 99999999


### --- ###
#0
def get_refined_xo2_full_egg_index_set_and_method_level_dicts(eggUnitIdxSets: dict) -> dict:
    eggUnitIdxSets = remove_subset_in_dict(eggUnitIdxSets)
    if "full" not in eggUnitIdxSets.keys():
        eggUnitIdxSets["full"] = get_all_index_set_from_dict(eggUnitIdxSets)
    eggMethodLevels = get_xo2_full_egg_method_level_dict(eggUnitIdxSets)
    return eggUnitIdxSets, eggMethodLevels

#0
def get_xo2_full_egg_method_level_dict(eggUnitIdxSets: dict) -> dict:
    eggMethodLevels = dict()
    for uName in eggUnitIdxSets.keys():
        eggMethodLevels[uName] = "H"
    eggMethodLevels["full"] = "L"
    return eggMethodLevels
### --- ###


### --- ###
#0
class XO2EggGenerator():
    def __init__(self, gjfFile: str, unitFile: str, rpaStrLst: list, 
                 sizeMode="r", eggFilePref="eggFile", maxAtom=LARGEINT, linkBonusMAtFile=""):
        self.GjfFile = gjfFile
        self.UnitFile = unitFile
        self.SizeMode = sizeMode
        self.RPALst = [float(rpa) for rpa in rpaStrLst]
        self.set_feature_list_for_selected_radius_mode() # Radius-Percentage-nAtom
        self.EggFilePref = eggFilePref
        self.MaxAtom = maxAtom
        self.LinkBonusMAtFile = linkBonusMAtFile
        self.U = DistUnit(unitFile, gjfFile, rpaLst=self.RPALst, radiusMode=sizeMode, maxAtom=maxAtom)
        self.UnitNeighborIdxSetLstsAsLevelLst = self.U.UnitNeighborIdxSetLstsAsLevelLst
    
    #0
    def set_feature_list_for_selected_radius_mode(self):
        # Radius-Percentage-nAtom
        if self.SizeMode == "r":
            self.RadiusLst = self.RPALst
        elif self.SizeMode == "p":
            self.NPercentLst = self.RPALst
        elif self.SizeMode == "a":
            self.NAtomLst = self.RPALst
        else:
            print(" Fatal error! Unknown unit size mode =", self.SizeMode)
            sys.exit(1)
    
    #0
    def get_egg_dicts_and_write_egg_file(self):
        if self.SizeMode == "r":
            eggIdxSetsPerRPA = self.U.get_egg_index_set_dict_for_each_radius()
        elif self.SizeMode == "p":
            eggIdxSetsPerRPA, radiusDictPerRPA = \
                    self.U.get_egg_index_set_dict_for_each_npercent(self.NPercentLst)
        elif self.SizeMode == "a":
            eggIdxSetsPerRPA, radiusDictPerRPA = \
                    self.U.get_egg_index_set_dict_for_each_natom(self.NAtomLst)
        else:
            print(" Fatal error! Unknown radius mode =", self.SizeMode)
            sys.exit(1)
        # Radius-Percentage-nAtom
        for uRPA in self.RPALst: 
            eggFileName = self.get_egg_file_name(uRPA)
            eggUnitIdxSets = self.get_good_egg_index_set_dict_from_raw_dict(eggIdxSetsPerRPA[uRPA])
            eggMethodLevels = get_xo2_full_egg_method_level_dict(eggUnitIdxSets)
            write_egg_file_from_egg_dicts(eggFileName, eggUnitIdxSets, eggMethodLevels)

    #1
    def get_good_egg_index_set_dict_from_raw_dict(self, rawEggIdxSets: dict) -> dict:
        eggUnitIdxSets = dict()
        for uid in sorted(rawEggIdxSets.keys()):
            iFragment = 0
            eggName = "U" + str(uid+1) + "F" + str(iFragment+1)
            eggUnitIdxSets[eggName] = rawEggIdxSets[uid]
        eggUnitIdxSets = remove_subset_in_dict(eggUnitIdxSets)
        allIdxSet = get_all_index_set_from_dict(eggUnitIdxSets)
        eggUnitIdxSets["full"] = allIdxSet
        return eggUnitIdxSets
    
    #1
    def get_egg_file_name(self, number: float) -> str:
        if self.MaxAtom == LARGEINT:
            if self.SizeMode == "r":
                eggFileName = "{0}.R{1:.1f}".format(self.EggFilePref, number)
            elif self.SizeMode == "p":
                eggFileName = "{0}.P{1:.1f}".format(self.EggFilePref, number)
            elif self.SizeMode == "a":
                eggFileName = "{0}.A{1:d}".format(self.EggFilePref, number)
            else:
                print(" Fatal error! Unknown radius mode =", self.SizeMode)
        else:
            if self.SizeMode == "r":
                eggFileName = "{0}.R{1:.1f}.M{2}".format(self.EggFilePref, number, self.MaxAtom)
            elif self.SizeMode == "p":
                eggFileName = "{0}.P{1:.1f}.M{2}".format(self.EggFilePref, number, self.MaxAtom)
            elif self.SizeMode == "a":
                eggFileName = "{0}.A{1:d}.M{2}".format(self.EggFilePref, number, self.MaxAtom)
            else:
                print(" Fatal error! Unknown radius mode =", self.SizeMode)
                sys.exit(1)
        if self.SizeMode == "p" or self.SizeMode == "a":
            digit = number - int(number*10) / 10
            if digit >= 0.01:
                print(" Warning! Precision loss of floatNum ({0}) in egg file name ({1}).".format(\
                      number, eggFileName))
        return eggFileName
### --- ###



### --- ###
#0
class XO2ManyBodyEggGenerator():
    def __init__(self, gjfFile: str, unitFile: str, radius2: float, radius3=0.0, radius4=0.0, 
                 eggFilePref="eggFile", maxAtom=LARGEINT, \
                 isAll3BodyPairDistanceBelow=True, isAll4BodyPairDistanceBelow=True):
        self.GjfFile = gjfFile
        self.UnitFile = unitFile
        self.Radius2 = radius2
        self.Radius3 = radius3
        self.Radius4 = radius4
        self.get_valid_radius_and_many_body_number_lists()
        self.EggFilePref = eggFilePref
        self.MaxAtom = maxAtom
        self.IsAll3BodyPairDistanceBelow = isAll3BodyPairDistanceBelow
        self.IsAll4BodyPairDistanceBelow = isAll4BodyPairDistanceBelow
        self.get_egg_file_name()
        self.DUnit = DistUnit(unitFile, gjfFile, rpaLst=self.ValidRadiusLst, radiusMode="r", maxAtom=maxAtom)
        self.get_high_level_egg_index_set_dict()
        self.EggIdxSets, self.EggMethodLevels = \
                get_refined_xo2_full_egg_index_set_and_method_level_dicts(self.HighEggIdxSets)
    
    #0
    def get_egg_file_name(self):
        R2Str, R3Str, R4Str = "", "", ""
        if self.Radius2 > 0:
            R2Str = ".R2_{0}".format(self.Radius2)
        if self.Radius3 > 0:
            if self.IsAll3BodyPairDistanceBelow:
                R3Str = ".R3A_{0}".format(self.Radius3) #A = all
            else:
                R3Str = ".R3C_{0}".format(self.Radius3) #C = center
        if self.Radius4 > 0:
            if self.IsAll4BodyPairDistanceBelow:
                R4Str = ".R4A_{0}".format(self.Radius4)
            else:
                R4Str = ".R4C_{0}".format(self.Radius4)
        self.EggFile = self.EggFilePref + R2Str + R3Str + R4Str + ".M" + str(self.MaxAtom)
    
    #0
    def get_valid_radius_and_many_body_number_lists(self):
        self.ValidRadiusLst, self.ValidManyBodyNumber = [], []
        for uRadius, iManyBody in zip([self.Radius2, self.Radius3, self.Radius4], [2,3,4]):
            if uRadius > 0:
                self.ValidRadiusLst.append(uRadius)
                self.ValidManyBodyNumber.append(iManyBody)
        if len(self.ValidRadiusLst) == 0:
            print("Fatal error!!! It is valid for all many body radii being ZERO or below.")
            sys.exit(1)
    
    #0
    def get_high_level_egg_index_set_dict(self):
        self.get_neighbor_index_set_list_dicts_per_many_body()
        self.get_egg_index_set_dict_from_neighbor_index_set_list_dicts()
    
    #1
    def get_neighbor_index_set_list_dicts_per_many_body(self):
        self.UnitNeighborIdxSetLstsPerManyBody = dict()
        unitNeighborIdxSetLstsAsLevelLst, _ = \
                self.DUnit.get_unit_neighbor_index_sets_in_dict_as_level_list_from_radius_list()
        self.EggIdxSetsPerManyBody = dict()
        for i, iManyBody in enumerate(self.ValidManyBodyNumber):
            self.UnitNeighborIdxSetLstsPerManyBody[iManyBody] = unitNeighborIdxSetLstsAsLevelLst[i]
    
    #1
    def get_egg_index_set_dict_from_neighbor_index_set_list_dicts(self):
        self.HighEggIdxSets = dict()
        for iManyBody in self.UnitNeighborIdxSetLstsPerManyBody.keys():
            if iManyBody == 2:
                self.update_with_two_body_egg_index_set_dict_with_limited_size(\
                        self.UnitNeighborIdxSetLstsPerManyBody[2])
            elif iManyBody == 3:
                self.update_with_three_body_egg_index_set_dict_with_limited_size(\
                        self.UnitNeighborIdxSetLstsPerManyBody[3])
            elif iManyBody == 4:
                self.update_with_four_body_egg_index_set_dict_with_limited_size(\
                        self.UnitNeighborIdxSetLstsPerManyBody[4])
            else:
                print("Fatal error!!! Invalid many body number =", iManyBody)
                sys.exit(1)
    
    #2
    def update_with_two_body_egg_index_set_dict_with_limited_size(self, unitNeighborIdxSetLsts: dict):
        for iUnit in sorted(unitNeighborIdxSetLsts.keys()):
            for jUnit in sorted(unitNeighborIdxSetLsts[iUnit][0]):
                if iUnit < jUnit:
                    unitLst = [iUnit, jUnit]
                    if self.DUnit.get_atom_number_from_unit_index_list(unitLst) <= self.MaxAtom:
                        eggName = self.get_egg_name(unitLst)
                        self.HighEggIdxSets[eggName] = set(unitLst)
    
    #2
    def update_with_three_body_egg_index_set_dict_with_limited_size(self, unitNeighborIdxSetLsts: dict):
        for iUnit in sorted(unitNeighborIdxSetLsts.keys()):
            for jUnit in sorted(unitNeighborIdxSetLsts[iUnit][0]):
                for kUnit in sorted(unitNeighborIdxSetLsts[iUnit][0]):
                    if iUnit < jUnit < kUnit:
                        unitLst = [iUnit, jUnit, kUnit]
                        if self.DUnit.get_atom_number_from_unit_index_list(unitLst) <= self.MaxAtom:
                            if self.IsAll3BodyPairDistanceBelow:
                                if kUnit in unitNeighborIdxSetLsts[jUnit][0]:
                                    eggName = self.get_egg_name(unitLst)
                                    self.HighEggIdxSets[eggName] = set(unitLst)
                            else:
                                eggName = self.get_egg_name(unitLst)
                                self.HighEggIdxSets[eggName] = set(unitLst)
                            
    #2
    def update_with_four_body_egg_index_set_dict_with_limited_size(self, unitNeighborIdxSetLsts: dict):
        for iUnit in sorted(unitNeighborIdxSetLsts.keys()):
            for jUnit in sorted(unitNeighborIdxSetLsts[iUnit][0]):
                for kUnit in sorted(unitNeighborIdxSetLsts[iUnit][0]):
                    for lUnit in sorted(unitNeighborIdxSetLsts[iUnit][0]):
                        if iUnit < jUnit < kUnit < lUnit:
                            unitLst = [iUnit, jUnit, kUnit, lUnit]
                            if self.DUnit.get_atom_number_from_unit_index_list(unitLst) <= self.MaxAtom:
                                if self.IsAll4BodyPairDistanceBelow:
                                    if kUnit in unitNeighborIdxSetLsts[jUnit][0] and \
                                            lUnit in unitNeighborIdxSetLsts[jUnit][0] and \
                                            lUnit in unitNeighborIdxSetLsts[kUnit][0]:
                                        eggName = self.get_egg_name(unitLst)
                                        self.HighEggIdxSets[eggName] = set(unitLst)
                                else:
                                    eggName = self.get_egg_name(unitLst)
                                    self.HighEggIdxSets[eggName] = set(unitLst)
    
    #3
    def get_egg_name(self, iUnitLst: list) -> str:
        eggName = ""
        for iUnit in iUnitLst:
            eggName += "U" + str(iUnit + 1)
        return eggName
        
    #0
    def write_egg_file(self):
        write_egg_file_from_egg_dicts(self.EggFile, self.EggIdxSets, self.EggMethodLevels)
### --- ###



### --- ###
class EggOptimizer():
    def __init__(self, gjfFile: str, unitFile: str, radius=2.0):
        self.GjfFile = gjfFile
        self.UnitFile = unitFile
        #
        self.RUnit = RadiusUnit(gjfFile, unitFile, radius)
        self.UnitIdLst = self.RUnit.UnitIdxLst
        #
        self.OutEggFilePrefix = "eggFile"
        self.OutUnitFilePrefix = "unitFile"
        self.TmpUnitFile = "tmpUnitFile"
    
    #0
    # link bonus = sum_of_force_error_after_bond - sum_of_force_error_before_bond
    # def initialize_bonus_matrix(self):
    #     self.LinkBonusPerPair = dict()
    #     for iUnit in self.RUnit.UnitIdxLst:
    #         self.LinkBonusPerPair[iUnit] = dict()
    #         for jUnit in self.RUnit.UnitIdxLst:
    #             self.LinkBonusPerPair[iUnit][jUnit] = 0.0
    
    #0
    def use_minimal_distance_as_unit_pair_2d_array(self):
        self.UnitPair2dArray = self.RUnit.MinDist2dArray
    
    #0
    def read_unit_pair_2d_array(self, unitPair2dArray: np.ndarray):
        self.UnitPair2dArray = unitPair2dArray
    
    #0
    def link_units_by_clustering_and_write_unit_and_egg_files(self, clusterMethod: str, \
                maxFragmentAtom: int, nRandom: int, pairDist: float, seed: int):
        self.MaxFragmentAtom = maxFragmentAtom
        self.NCluster = ceil(self.RUnit.U.TotAtom / maxFragmentAtom)
        self.ClusterMethod = clusterMethod
        if self.NCluster > self.RUnit.U.N:
            print("Warning!!! Too small maximal fragment atom number and not enough units. \
                  No further clustering is needed.")
            self.get_output_unit_file_name("no-clustering", seed=seed)
            write_unit_file(self.OutUnitFile, self.RUnit.U.TotAtom, self.RUnit.U.AtomIdxUnitLst, \
                            self.RUnit.U.UnitIdxLst, self.RUnit.U.UnitChargeLst, self.RUnit.U.UnitSpinLst)
            self.get_output_egg_file_name("no-clustering", seed=seed)
        else:
            for iRandom in range(1, nRandom+1):
                tmpAtomIdxUnitLst, tmpUnitIdxLst, tmpUnitChargeLst, tmpUnitSpinLst, clusterSum \
                        = self.get_merged_units_lists_by_clustering()
                self.get_output_unit_file_name(clusterMethod, clusterSum, iRandom, seed=seed)
                write_unit_file(self.OutUnitFile, self.RUnit.U.TotAtom, tmpAtomIdxUnitLst, \
                                tmpUnitIdxLst, tmpUnitChargeLst, tmpUnitSpinLst)
                self.get_output_egg_file_name(clusterMethod, clusterSum, iRandom, seed=seed)
        eggGenerator = XO2EggGenerator(self.GjfFile, self.OutUnitFile, [pairDist], 
                                       sizeMode="r", eggFilePref=self.OutEggFile)
        eggGenerator.get_egg_dicts_and_write_egg_file()
    
    #1
    def get_merged_units_lists_by_clustering(self):
        clusterSum = None
        if self.ClusterMethod == "balanced-k-means":
            unitNAtomArray = self.get_unit_atom_number_array()
            unitIdxLstPerCluster, centerIdLst = selective_balanced_k_means_for_distance_2d_array(\
                    self.UnitPair2dArray, self.RUnit.MinDist2dArray, self.NCluster, 
                    self.MaxFragmentAtom, unitNAtomArray, selectiveMethod="k_centers", isRandom=True)
            clusterSum = get_cluster_sum(self.UnitPair2dArray, centerIdLst, unitIdxLstPerCluster)
        elif self.ClusterMethod == "k-centers":
            unitIdxLstPerCluster, _ = k_centers_for_distance_2d_array(\
                    self.UnitPair2dArray, self.NCluster)
        elif self.ClusterMethod == "k-means":
            unitIdxLstPerCluster, centerIdLst = k_means_for_distance_2d_array(\
                    self.UnitPair2dArray, self.NCluster)
            clusterSum = get_cluster_sum(self.UnitPair2dArray, centerIdLst, unitIdxLstPerCluster)
        else:
            print("Fatal error! Unsupported clustering method =", self.ClusterMethod)
            sys.exit(1)
        tmpAtomIdxUnitLst, tmpUnitIdxLst, tmpUnitChargeLst, tmpUnitSpinLst = \
                self.get_linked_unit_lists(unitIdxLstPerCluster)
        return tmpAtomIdxUnitLst, tmpUnitIdxLst, tmpUnitChargeLst, tmpUnitSpinLst, clusterSum
    
    #2
    def get_unit_atom_number_array(self) -> None:
        unitNAtomLst = [len(self.RUnit.AtomIdxUnitLst[iUnit]) for iUnit in self.UnitIdLst]
        return np.array(unitNAtomLst)
    
    #1
    def get_output_unit_file_name(self, clusterMethod: str, clusterSum=None, iRandom=0, seed=-1):
        clusterStr = self.get_cluster_string(clusterMethod, clusterSum)
        fragmentStr = "_f" + str(self.MaxFragmentAtom)
        randStr = self.get_random_string(iRandom)
        seedStr = "_seed" + str(seed)
        self.OutUnitFile = self.OutUnitFilePrefix + clusterStr + fragmentStr + randStr + seedStr
    
    #2
    def get_cluster_string(self, clusterMethod: str, clusterSum: float) -> str:
        if clusterSum is None:
            return "_" + clusterMethod
        else:
            return "_{0}={1:.5e}".format(clusterMethod, clusterSum)
    
    #2
    def get_random_string(self, iRandom: int) -> str:
        if iRandom == 0:
            return ""
        else:
            return "_rand" + str(iRandom)

    #1
    def get_output_egg_file_name(self, clusterMethod: str, clusterSum=None, iRandom=0, pairDist=None, seed=-1):
        clusterStr = self.get_cluster_string(clusterMethod, clusterSum)
        fragmentStr = "_f" + str(self.MaxFragmentAtom)
        randStr = self.get_random_string(iRandom)
        seedStr = "_seed" + str(seed)
        self.OutEggFile = self.OutEggFilePrefix + clusterStr + fragmentStr + randStr + seedStr
    
    
        
    # #2
    # def get_unit_object_after_link_and_write_old(self, unitFile: str):
    #     linkedUnitSets = pair_lists_to_set_dict(self.LinkedUnitPairIdxLst)
    #     atomIdxLstOfUnitLst, unitIdxLst, unitChargeLst, unitSpinLst = \
    #             self.get_linked_unit_lists(linkedUnitSets)
    #     write_unit_file(unitFile, self.Unit.TotAtom, atomIdxLstOfUnitLst, \
    #                     unitIdxLst, unitChargeLst, unitSpinLst)
    
    #3
    def get_linked_unit_lists(self, linkedUnitSets: dict) -> tuple:
        atomIdxLstOfUnits, unitCharges, unitSpins = self.get_unit_dicts_from_unit_object()
        for iUnit in linkedUnitSets.keys():
            atomIdxLst, sumCharge, sumSpin = [], 0, 1
            for jUnit in linkedUnitSets[iUnit]:
                atomIdxLst += atomIdxLstOfUnits[jUnit]
                sumCharge += unitCharges[jUnit]
                sumSpin += unitSpins[jUnit] - 1
            for jUnit in linkedUnitSets[iUnit]:
                del atomIdxLstOfUnits[jUnit]
                del unitCharges[jUnit]
                del unitSpins[jUnit]
            atomIdxLstOfUnits[iUnit] = atomIdxLst
            unitCharges[iUnit] = sumCharge
            unitSpins[iUnit] = sumSpin
        atomIdxLstOfUnitLst = dict_to_lists_in_list(atomIdxLstOfUnits)
        unitIdxLst = list(range(len(atomIdxLstOfUnitLst)))
        unitChargeLst = dict_to_lists_in_list(unitCharges)
        unitSpinLst = dict_to_lists_in_list(unitSpins)
        return atomIdxLstOfUnitLst, unitIdxLst, unitChargeLst, unitSpinLst
    
    #4
    def get_unit_dicts_from_unit_object(self) -> tuple:
        atomIdxLstOfUnits, unitCharges, unitSpins = dict(), dict(), dict()
        for iUnit in self.RUnit.UnitIdxLst:
            atomIdxLstOfUnits[iUnit] = deepcopy(self.RUnit.AtomIdxUnitLst[iUnit])
            unitCharges[iUnit] = self.RUnit.UnitChargeLst[iUnit]
            unitSpins[iUnit] = self.RUnit.UnitSpinLst[iUnit]
        return atomIdxLstOfUnits, unitCharges, unitSpins
    
    #2
    def get_atom_number_of_maximal_egg(self, unitObj, eggUnitIdxSets: dict):
        maxHvAtm, maxAtom = unitObj.get_max_atom_number_among_eggs(eggUnitIdxSets)
        return maxAtom
    
    #0
    def link_units_with_greedy_method_and_write_unit_and_egg_files(self, maxFragmentAtom: int, pairDist: float):
        self.MaxFragmentAtom = maxFragmentAtom
        self.link_unit_with_greedy_method_and_write()
        self.get_egg_dicts_from_new_unit_lists_and_write(pairDist)
    
    #1
    def link_unit_with_greedy_method_and_write(self):
        self.link_units_with_greedy_method()
        self.get_output_unit_file_name("greedy")
        write_unit_file(self.OutUnitFile, self.Unit.TotAtom, self.NewAtomIdxLstOfUnitLst, \
                        self.NewUnitIdxLst, self.NewUnitChargeLst, self.NewUnitSpinLst)
    
    #2
    def link_units_with_greedy_method(self):
        tempUnitFile = "temp_unitfile"
        self.get_link_pair_id_and_link_bonus_lists()
        self.LinkedUnitPairIdxLst = []
        pairIdxLst = np.argsort(self.LinkBonusLst)
        for pairIdx in pairIdxLst:
            self.LinkedUnitPairIdxLst.append(self.LinkPairIdLst[pairIdx])
            self.get_unit_object_after_link_and_write(tempUnitFile)
            tempUnitObj = Unit(tempUnitFile)
            if self.get_atom_number_of_maximal_egg(tempUnitObj, self.eggUnitIdxSets) > self.MaxFragmentAtom:
                del self.LinkedUnitPairIdxLst[-1]
    
    #3
    def get_link_pair_id_and_link_bonus_lists(self):
        self.LinkBonusLst, self.LinkPairIdLst = [], []
        for iUnit in range(self.Unit.N):
            for jUnit in range(self.Unit.N):
                self.LinkBonusLst.append(self.UnitPair2dArray[iUnit][jUnit])
                self.LinkPairIdLst.append([iUnit, jUnit])
    
    #3
    def get_unit_object_after_link_and_write(self, unitFile: str):
        linkedUnitSets = pair_lists_to_set_dict(self.LinkedUnitPairIdxLst)
        atomIdxLstOfUnitLst, unitIdxLst, unitChargeLst, unitSpinLst = \
                self.get_linked_unit_lists(linkedUnitSets)
        self.NewAtomIdxLstOfUnitLst = atomIdxLstOfUnitLst
        self.NewUnitIdxLst = unitIdxLst
        self.NewUnitChargeLst = unitChargeLst
        self.NewUnitSpinLst = unitSpinLst
        write_unit_file(unitFile, self.Unit.TotAtom, atomIdxLstOfUnitLst, \
                        unitIdxLst, unitChargeLst, unitSpinLst)
    
    #4
    def get_linked_unit_lists(self, linkedUnitSets: dict) -> tuple:
        atomIdxLstOfUnits, unitCharges, unitSpins = dict(), dict(), dict()
        for iUnit in self.RUnit.UnitIdxLst:
            atomIdxLstOfUnits[iUnit] = deepcopy(self.RUnit.AtomIdxUnitLst[iUnit])
            unitCharges[iUnit] = self.RUnit.UnitChargeLst[iUnit]
            unitSpins[iUnit] = self.RUnit.UnitSpinLst[iUnit]
        for iUnit in linkedUnitSets.keys():
            atomIdxLst, sumCharge, sumSpin = [], 0, 1
            for jUnit in linkedUnitSets[iUnit]:
                atomIdxLst += atomIdxLstOfUnits[jUnit]
                sumCharge += unitCharges[jUnit]
                sumSpin += unitSpins[jUnit] - 1
            for jUnit in linkedUnitSets[iUnit]:
                del atomIdxLstOfUnits[jUnit]
                del unitCharges[jUnit]
                del unitSpins[jUnit]
            atomIdxLstOfUnits[iUnit] = atomIdxLst
            unitCharges[iUnit] = sumCharge
            unitSpins[iUnit] = sumSpin
        atomIdxLstOfUnitLst = dict_to_lists_in_list(atomIdxLstOfUnits)
        unitIdxLst = list(range(len(atomIdxLstOfUnitLst)))
        unitChargeLst = dict_to_lists_in_list(unitCharges)
        unitSpinLst = dict_to_lists_in_list(unitSpins)
        return atomIdxLstOfUnitLst, unitIdxLst, unitChargeLst, unitSpinLst
    
    #1
    def get_egg_dicts_from_new_unit_lists_and_write(self, pairDist: float):
        self.get_output_egg_file_name("greedy")
        eggGenerator = XO2EggGenerator(self.GjfFile, self.OutUnitFile, [pairDist], 
                                       sizeMode="r", eggFilePref=self.OutEggFile)
        eggGenerator.get_egg_dicts_and_write_egg_file()
### --- ###



