import os
import sys
import re
import numpy as np
from shutil import copy, rmtree
from copy import deepcopy
from lib.unit import Unit, DistUnit, RadiusUnit
from lib.capper import UnitFragmentCapper, FragmentCapper
from lib.QMManager import QMManager
from lib.misc import flatten, backup_serial_file, mk_serial_dir
from lib.misc import compress_continous_index_into_serial_string
from lib.misc import get_all_index_set_from_dict, remove_subset_in_dict
from lib.egg import get_xo2_full_egg_method_level_dict
from lib.egg import get_refined_xo2_full_egg_index_set_and_method_level_dicts
from lib.misc import get_dir_name_from_index_list
from lib.misc import get_method_level_and_index_list_from_calc_path
from lib.misc import get_method_level_from_unit_or_work_path
from lib.misc import print_error_and_exit
from lib.misc import get_center_unit_index_from_egg_name, get_full_index_set_from_dict
from file.gaussian import read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file
from file.gaussian import read_genecp_file
from file.gaussian import get_option_from_route_line
from file.gaussian import renew_gjf_file_with_standard_orientation
from file.gaussian import write_gaussian_external_output_file
from file.gaussian import ensure_consistency_between_unit_file_and_gjf_file
from file.other import read_global_index_file, write_global_index_file
from file.other import is_valid_connectivity_file
from file.other import read_weight_file, write_weight_file
from file.other import read_simple_XO2_fragment_set_and_level_dict_from_scheme_file
from file.other import read_unit_egg_files_and_output_egg_dicts
from file.other import append_path_alias_file_and_return_new_path_alias_dict, read_path_alias_file
from file.other import get_egg_name_and_index_set_lists_at_selected_level
from file.other import print_xo_log, print_section_to_xo_log
from file.xo import write_egg_file_from_egg_dicts
from file.xo import read_data_from_config_file
from file.sum import get_result_type_from_input_file
from file.unit import read_unit_file, renew_additional_unit_file_and_return_index_of_new_unit


class XoGeneral():
    def __init__(self):
        # constant paramenters
        self.GjfFile = "standard.gjf"
        self.ConfigFile = "configfile"
        self.EggFile = "eggfile"
        self.ExternalEggFile = self.EggFile
        self.EggAtomFile = "eggatomfile"
        self.WeightFile = "weightfile"
        self.PairWeightFile = "pairweightfile"
        self.FragmentFilePref = "fragment"
        self.FullFilePref = "full"
        self.GlobalIdxFile = "global_index"
        self.PathAliasFile = "pathaliasfile"
        self.GenecpFiles = {"H": "genecp_high", "M": "genecp_medium", "L": "genecp_low"}
        # self.U.UnitNeighborhoodSets[iLevel][iUnit] is a list of sets as iUnit may center multiple fragments
        # constant properties
        self.ValidNeighborTypeLst = ["D", "C"]
        self.ValidErrorTypeLst = ["F"] # "C"
        self.ValidUpdateModeLst = ["LXM", "LEVELEXPANDMERGE", 
                                   "UXCM", "UNITEXPANDCENTERMERGE", # LXM
                                   "LXCM", "LEVELEXPANDCENTERMERGE", # LWXLM
                                   "UXNM", "UNITEXPANDNEIGHBORMERGE", # LXLNM
                                   "LXNM", "LEVELEXPANDNEIGHBORMERGE", # UXCM
                                   "SXM", "SIZEEXPANDMERGE", 
                                   "UX", "UNITEXPAND", 
                                   "LX", "LEVELEXPAND"]
        # self.ErrorType = "F"
        self.ValidMethodLevelSet = set(["H", "M", "L"])
        # obtained properties #
        self.Methods, self.ExecutableCommands, self.Executables, self.GaussianExtraInputs \
                = read_data_from_config_file(self.ConfigFile)
        self.MethodLevelLst = sorted(self.Methods.keys())
        self.GetWeightFileCommand = "/share/home/jwweng/g16XO/get_weight_file"

    ##### ----- #####
    #0
    def set_external_egg_file(self, extEggFile: str):
        self.ExternalEggFile = extEggFile

    #0
    def set_egg_unit_idx_set_dict(self, eggIdxSets: dict):
        self.EggIdxSets = eggIdxSets

    #0
    def set_egg_method_level_dict(self, eggMethodLevels: dict):
        self.EggMethodLevels = eggMethodLevels
    ##### ----- #####
    
    ##### ----- #####
    #0
    def get_valid_neighbor_type(self, neighborType: str) -> str:
        if neighborType not in self.ValidNeighborTypeLst:
            print_error_and_exit("Invalid neighbor type =", neighborType)
        else:
            return neighborType

    def get_valid_error_type(self, errorType: str) -> str:
        if errorType not in self.ValidErrorTypeLst:
            print_error_and_exit("Invalid partitioner type =", errorType)
        else:
            return errorType
    ##### ----- #####

    ##### ----- #####
    def read_basis_set_and_ecp_from_genecp_file(self) -> tuple:
        basisSets, ecps = dict(), dict()
        for level in self.MethodLevelLst:
            basisSets[level], ecps[level] = read_genecp_file(self.GenecpFiles[level])
            if len(basisSets[level]) != 0:
                print("------------------------------- genecp -------------------------------")
                print(" - level", level)
                print("   - basis sets:")
                for element in sorted(basisSets[level].keys()):
                    print("   -", element, ":", basisSets[level][element])
                print("   - ecps:")
                for element in sorted(ecps[level].keys()):
                    print("   -", element, ":", ecps[level][element])
                print("----------------------------------------------------------------------")
        return basisSets, ecps
    ##### ----- #####

    ##### ----- #####
    def get_mem_from_nproc(self, nProc: str) -> str:
        nProc = int(nProc)
        if nProc > 35:
            mem = nProc * 7
            return str(mem)+"GB"
        elif nProc > 27:
            mem = int(nProc * 5)
            return str(mem)+"GB"
        elif nProc > 20:
            mem = int(nProc * 1.5)
            return str(mem)+"GB"
        else:
            return "2000MW"
    ##### ----- #####
    
    ##### ----- #####
    def get_xo_option_from_route_line(self, routeLine: str) -> str:
        xoOption = get_option_from_route_line(routeLine)
        if not re.search("nosymm", xoOption):
            xoOption += " nosymm"
        if not re.search("conn", xoOption):
            xoOption += " geom=connectivity"
        return xoOption
    ##### ----- #####
    
    ##### ----- #####
    def get_standard_orientation(self, inGjfFile: str, outGjfFile: str, QmManager, useStandardOrientation=True):
        if not useStandardOrientation:
            copy(inGjfFile, outGjfFile)
        else:
            if QmManager.ProgramTypes["H"] == "gaussian":
                renew_gjf_file_with_standard_orientation(inGjfFile, outGjfFile, QmManager.Executables["H"])
            else:
                # print_xo_log("Fail to standardize orientation as Gaussian program is not at high-level,", 
                #                      "but {0} is given.".format(QmManager.ProgramTypes["H"]))
                print_xo_log(" WARNING! Did not standardize orientation as Gaussian program is not at high-level.")
                copy(inGjfFile, outGjfFile)
    ##### ----- #####

    ##### ----- #####
    def ensure_valid_unique_level_set(self, uniqueLevelSet: set) -> None:
        if not uniqueLevelSet <= self.ValidMethodLevelSet:
            print_error_and_exit("Invalid level =", ", ".join(uniqueLevelSet), ". Valid ones are:", 
                    ", ".join(self.ValidMethodLevelSet))
        if len(uniqueLevelSet) == 3:
            if uniqueLevelSet != self.ValidMethodLevelSet:
                print_error_and_exit("XO3 must have level H, M and L, not {0}, {1} and {2}".format(\
                        uniqueLevelSet[0], uniqueLevelSet[1], uniqueLevelSet[2]))
        elif len(uniqueLevelSet) == 2:
            if uniqueLevelSet != set(["H", "L"]):
                print_error_and_exit("XO2 must have level H and L, not", uniqueLevelSet[0], "and", uniqueLevelSet[1])
        elif len(uniqueLevelSet) == 1:
            if uniqueLevelSet != set(["H"]):
                print_error_and_exit("XO1 must have level H only, not", uniqueLevelSet[0])
        else:
            print_error_and_exit("Only 1, 2, or 3 level(s) are allowed, not {0}. Check egg file.".format())
    ##### ----- #####
    
    ##### ----- #####
    #0
    def set_weight_file(self, weightFile: str):
        self.WeightFile = weightFile
    ##### ----- #####
    
    ##### ----- #####
    def refine_nproc_with_nProc(self, nProc: str) -> str:
        if nProc < 0:
            return 1
        else:
            try:
                return int(nProc)
            except ValueError:
                print_error_and_exit("Invalid number of nproc =", nProc)
    
    def refine_mem_with_nProc(self, mem: str) -> str:
        if mem=="" and int(self.Nproc) != 0:
            return self.get_mem_from_nproc(self.Nproc)
        else:
            return mem
    ##### ----- #####


class XoSetup(XoGeneral):
    def __init__(self, gjfFile: str, unitFile: str, connectivityFile: str, fragmentDir: str, nProc: str, \
                 eggLevelArray: np.ndarray, neighborType="C", connectivityCutoff=0.002, \
                 initRadius=3.0, incrRadius=2.0, deltaRadius=0.1, maxIncrHvyAtom=25, nEggLevel=3, \
                 radiusLst=[], backChargeRange=0.0, backChargeFile="", oldWeightFile="", mem=""):
        XoGeneral.__init__(self)
        # input parameters
        self.GjfFile = gjfFile
        self.UnitFile = unitFile
        self.ConnectivityFile = connectivityFile
        self.FragmentDir = fragmentDir
        self.NProc = str(nProc)
        self.EggLevelArray = eggLevelArray
        self.NeighborType = self.get_valid_neighbor_type(neighborType)
        self.ConnectivityCutoff = connectivityCutoff
        self.InitRadius = initRadius
        self.IncrRadius = incrRadius
        self.DeltaRadius = deltaRadius
        self.MaxIncrHvyAtom = maxIncrHvyAtom
        self.NEggLevel = nEggLevel
        self.MaxEggLevel = nEggLevel - 1
        self.RadiusLst = radiusLst
        self.BackChargeRange = backChargeRange
        self.BackChargeFile = backChargeFile
        self.OldWeightFile = oldWeightFile
        # constant properties
        # derived properties
        self.ElementSpins = dict()
        ensure_consistency_between_unit_file_and_gjf_file(unitFile, gjfFile)
        self.Mem = self.refine_mem_with_nProc(mem)
        self.U = self.get_unit_object()
        self.UnitNeighborIdxSetLstsAsLevelLst = self.U.UnitNeighborIdxSetLstsAsLevelLst
        self.Capper = UnitFragmentCapper(gjfFile, unitFile, backChargeRange, backChargeFile)
        self.Charge, self.Spin, self.Xyz, self.ElementLst, self.BondOrders = \
                read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(gjfFile)
        self.BasisSets, self.Ecps = self.read_basis_set_and_ecp_from_genecp_file()
        self.ResultType = get_result_type_from_input_file(gjfFile)
        self.TotAtom, self.AtomIdxLstOfUnitLst, self.UnitIdxLst, self.UnitChargeLst, self.UnitSpinLst = \
                read_unit_file(unitFile)
        self.QmManager = QMManager(self.ConfigFile, self.GjfFile, nProc=nProc)
    
    def get_unit_object(self):
        if self.NeighborType[0] == "C":
            if not is_valid_connectivity_file(self.ConnectivityFile):
                print_error_and_exit("No valid connectivity file is found.", \
                                     "Call ConnectivityCalculator to get the connectivity file.")
            print_xo_log(" Building unit neighborhood from connectivity file =", self.ConnectivityFile, "...")
            return Unit(self.UnitFile, self.NeighborType, self.ConnectivityCutoff, self.ConnectivityFile)
        elif self.NeighborType == "D": 
            return DistUnit(self.UnitFile, self.GjfFile, self.InitRadius, self.IncrRadius, \
                            self.DeltaRadius, self.MaxIncrHvyAtom, self.NEggLevel, self.RadiusLst)
        #TODO return RadiusUnit(...)
    
    ##### ----- #####
    #####
    def write_xo2_full_egg_file_from_egg_level_array_and_merge_dict(self, 
            eggLevelArray: np.ndarray, mergeIdxLsts: dict) -> tuple: # -> unit_egg_file
        self.EggIdxSets, self.EggMethodLevels = \
                self.get_refined_xo2_full_egg_dicts_from_egg_level_array_and_merge_dict(eggLevelArray, mergeIdxLsts)
        self.write_egg_file_from_egg_dicts_and_backup()
    
    ####
    def get_refined_xo2_full_egg_dicts_from_egg_level_array_and_merge_dict(self, eggLevelArray: np.ndarray, 
                                                                           mergeIdxLsts: dict) -> dict:
        eggUnitIdxSets = self.get_refined_xo2_full_egg_unit_index_set_dict_from_egg_level_array_and_merge_dict(\
                eggLevelArray, mergeIdxLsts)
        eggMethodLevels = get_xo2_full_egg_method_level_dict(eggUnitIdxSets)
        return eggUnitIdxSets, eggMethodLevels
    
    ###
    def get_refined_xo2_full_egg_unit_index_set_dict_from_egg_level_array_and_merge_dict(self, 
            eggLevelArray: np.ndarray, mergeIdxLsts: dict) -> dict:
        eggUnitIdxSets = dict()
        fullEggUnitIdxSetLsts = self.get_egg_unit_index_set_from_egg_level_array_and_merge_dict(eggLevelArray, 
                                                                                                mergeIdxLsts)
        for uid in sorted(fullEggUnitIdxSetLsts.keys()):
            for iFragment, eggUnitIdxSet in enumerate(fullEggUnitIdxSetLsts[uid]):
                eggName = "U" + str(uid+1) + "F" + str(iFragment+1)
                eggUnitIdxSets[eggName] = eggUnitIdxSet
        eggUnitIdxSets = remove_subset_in_dict(eggUnitIdxSets)
        allIdxSet = get_all_index_set_from_dict(eggUnitIdxSets)
        eggUnitIdxSets["full"] = allIdxSet
        return eggUnitIdxSets

    ##
    def get_egg_unit_index_set_from_egg_level_array_and_merge_dict(self, eggLevelArray: np.ndarray, 
                                                                   mergeIdxLsts: dict) -> dict:
        fullEggUnitIdxSetLsts = dict()
        for centerUnitIdx, uLevel in enumerate(eggLevelArray):
            fullEggUnitIdxSetLsts[centerUnitIdx] = []
            for fullEggUnitIdxSet in self.UnitNeighborIdxSetLstsAsLevelLst[uLevel][centerUnitIdx]:
                newSet = self.add_merged_unit_to_egg(fullEggUnitIdxSet, mergeIdxLsts)
                fullEggUnitIdxSetLsts[centerUnitIdx].append(newSet)
        return fullEggUnitIdxSetLsts
    
    #
    def add_merged_unit_to_egg(self, fullEggUnitIdxSet: set, mergeIdxLsts: dict) -> set:
        newLst = []
        for centerUnitIdx in fullEggUnitIdxSet:
            newLst.extend(mergeIdxLsts[centerUnitIdx] + [centerUnitIdx])
        return set(newLst)
    
    #####
    def write_xo2_full_egg_file_from_egg_level_array(self) -> tuple: # -> unit_egg_file
        self.EggIdxSets, self.EggMethodLevels = self.get_xo2_full_egg_dicts_from_egg_level_array()
        self.write_egg_file_from_egg_dicts_and_backup()
        # return eggIdxSets, eggMethodLevels
    
    ####
    def get_xo2_full_egg_dicts_from_egg_level_array(self) -> tuple:
        eggUnitIdxSets = self.get_xo2_full_egg_unit_index_set_dict_from_egg_level_array()
        eggMethodLevels = get_xo2_full_egg_method_level_dict(eggUnitIdxSets)
        return eggUnitIdxSets, eggMethodLevels

    ###
    def get_xo2_full_egg_unit_index_set_dict_from_egg_level_array(self) -> dict:
        eggUnitIdxSets = dict()
        fullEggUnitIdxSetLsts = self.get_egg_unit_index_set_from_egg_level_array()
        for uid in sorted(fullEggUnitIdxSetLsts.keys()):
            for iFragment, eggUnitIdxSet in enumerate(fullEggUnitIdxSetLsts[uid]):
                eggName = "U" + str(uid+1) + "F" + str(iFragment)
                eggUnitIdxSets[eggName] = eggUnitIdxSet
        eggUnitIdxSets = remove_subset_in_dict(eggUnitIdxSets)
        allIdxSet = get_all_index_set_from_dict(eggUnitIdxSets)
        eggUnitIdxSets["full"] = allIdxSet
        return eggUnitIdxSets
    
    #
    def get_egg_unit_index_set_from_egg_level_array(self) -> dict:
        fullEggUnitIdxSetLsts = dict()
        for unitIdx, uLevel in enumerate(self.EggLevelArray):
            fullEggUnitIdxSetLsts[unitIdx] = self.UnitNeighborIdxSetLstsAsLevelLst[uLevel][unitIdx]
        return fullEggUnitIdxSetLsts
    
    ####
    def write_egg_file_from_egg_dicts_and_backup(self):
        backup_serial_file(self.EggFile)
        self.write_egg_file_from_egg_dicts()
    
    ###
    def write_egg_file_from_egg_dicts(self):
        nameLst = sorted(self.EggIdxSets.keys())
        with open(self.EggFile, "w") as wfl:
            print("[eggs]", file=wfl)
            for uName in nameLst:
                print("{0:10s}:".format(uName), 
                        compress_continous_index_into_serial_string(self.EggIdxSets[uName], ","), file=wfl)
            print(file=wfl)
            print("[levels]", file=wfl)
            for uName in nameLst:
                print("{0:10s}: {1}".format(uName, self.EggMethodLevels[uName]), file=wfl)
            print(file=wfl)
   
    ##### ----- #####
    
    ##### ----- #####
    #0
    def get_fragment_weight_and_dir_lists_and_write_to_weight_and_path_alias_files(self):
        self.AllIdxSetAtLevels, self.HighEggIdxSets, self.MediumEggIdxSets, self.LowEggIdxSets, \
                self.UniqueLevelLst = self.get_index_set_at_each_level()
        self.FragmentDir1stChar = "U"
        fragmentPathLst = self.write_weight_file_and_return_fragment_path_list()
        _ = append_path_alias_file_and_return_new_path_alias_dict(self.PathAliasFile, fragmentPathLst)
    ##### ----- #####
    
    ##### ----- #####
    #0-
    def get_index_set_at_each_level(self) -> dict:
        uniqueLevelLst = self.get_valid_unique_level_list()
        allIdxSetAtLevels = self.get_all_index_set_at_each_level(uniqueLevelLst)
        highEggIdxSets, mediumEggIdxSets, lowEggIdxSets = self.get_egg_index_set_at_each_level_without_subset()
        return allIdxSetAtLevels, highEggIdxSets, mediumEggIdxSets, lowEggIdxSets, uniqueLevelLst
    
    #1
    def get_valid_unique_level_list(self) -> list:
        uniqueLevelSet = set(self.EggMethodLevels.values())
        self.ensure_valid_unique_level_set(uniqueLevelSet)
        return sorted(uniqueLevelSet)
    
    #1
    def get_all_index_set_at_each_level(self, uniqueLevelLst: list) -> list:
        allIdxSetAtLevels = dict()
        for level in uniqueLevelLst:
            allIdxSetAtLevels[level] = set([])
        for uName in self.EggIdxSets.keys():
            eggMethodLevel = self.EggMethodLevels[uName]
            allIdxSetAtLevels[eggMethodLevel] |= self.EggIdxSets[uName]
        return allIdxSetAtLevels

    #1
    def get_egg_index_set_at_each_level_without_subset(self) -> tuple:
        highEggIdxSets, mediumEggIdxSets, lowEggIdxSets = self.get_egg_index_set_at_each_level()
        highEggIdxSets = remove_subset_in_dict(highEggIdxSets) 
        mediumEggIdxSets = remove_subset_in_dict(mediumEggIdxSets) 
        lowEggIdxSets = remove_subset_in_dict(lowEggIdxSets) 
        return highEggIdxSets, mediumEggIdxSets, lowEggIdxSets
    
    #2
    def get_egg_index_set_at_each_level(self) -> tuple:
        highEggIdxSets, mediumEggIdxSets, lowEggIdxSets = dict(), dict(), dict()
        for uName in self.EggIdxSets.keys():
            eggMethodLevel = self.EggMethodLevels[uName]
            if eggMethodLevel == "H":
                highEggIdxSets[uName] = self.EggIdxSets[uName]
            elif eggMethodLevel == "M":
                mediumEggIdxSets[uName] = self.EggIdxSets[uName]
            elif eggMethodLevel == "L":
                lowEggIdxSets[uName] = self.EggIdxSets[uName]
            else:
                print_error_and_exit("Invalid level =", eggMethodLevel)
        return highEggIdxSets, mediumEggIdxSets, lowEggIdxSets
    ##### ----- #####

    ##### ----- #####
    #0
    def write_weight_file_and_return_fragment_path_list(self):
        if "OldWeightFile" in self.__dict__ and os.path.isfile(self.OldWeightFile):
            if self.OldWeightFile != self.WeightFile:
                copy(self.OldWeightFile, self.WeightFile)
                print_xo_log(" Using existing weight file =", self.OldWeightFile)
                fragmentWeightLst, fragmentPathLst = read_weight_file(self.WeightFile)
        else: #TODO
            if False and os.path.isfile(self.GetWeightFileCommand) and os.path.isfile(self.ExternalEggFile):
                print_xo_log(" Calculating weight file with RUST code:")
                tmpCmd = self.GetWeightFileCommand + " " + self.ExternalEggFile + " " + self.WeightFile \
                         + " " + self.FragmentDir + " " + self.FragmentDir1stChar
                print_xo_log(" - " + tmpCmd)
                read = os.popen(tmpCmd).read()
                print_xo_log(read)
                fragmentWeightLst, fragmentPathLst = read_weight_file(self.WeightFile)
            else:
                print_xo_log(" Calculating weight file with python code ...")
                fragmentWeightLst, fragmentPathLst = self.derive_fragment_weight_and_path_lists()
                write_weight_file(self.WeightFile, fragmentWeightLst, fragmentPathLst)
        return fragmentPathLst
    
    #1
    def derive_fragment_weight_and_path_lists(self) -> list:
        fragmentWeightLst, fragmentPathLst = [], []
        if len(self.UniqueLevelLst) == 1:
            fragmentWeightLst, fragmentPathLst, fragmentNameLst = \
                    self.get_weight_path_and_name_lists_of_derived_fragments("H")
        elif len(self.UniqueLevelLst) == 2:
            weightLst1, pathLst1, nameLst1 = \
                    self.get_weight_path_and_name_lists_of_derived_fragments("L")
            weightLst2, pathLst2, nameLst2 = \
                    self.get_weight_path_and_name_lists_of_derived_fragments_with_fore_and_back_grounds("H", "L")
            fragmentWeightLst = weightLst1 + weightLst2
            fragmentPathLst = pathLst1 + pathLst2
        elif len(self.UniqueLevelLst) == 3:
            weightLst1, pathLst1, nameLst1 = \
                    self.get_weight_path_and_name_lists_of_derived_fragments("L")
            weightLst2, pathLst2, nameLst2 = \
                    self.get_weight_path_and_name_lists_of_derived_fragments_with_fore_and_back_grounds("M", "L")
            weightLst3, pathLst3, nameLst3 = \
                    self.get_weight_path_and_name_lists_of_derived_fragments_with_fore_and_back_grounds("H", "M")
            fragmentWeightLst = weightLst1 + weightLst2 + weightLst3
            fragmentPathLst = pathLst1 + pathLst2 + pathLst3
        else:
            print_error_and_exit("Invalid number of levels =", len(self.UniqueLevelLst))
        return fragmentWeightLst, fragmentPathLst
    
    #2
    def get_weight_path_and_name_lists_of_derived_fragments(self, selLevel: str) -> tuple:
        eggNameLst, eggIdxSetInLst = get_egg_name_and_index_set_lists_at_selected_level(self.EggIdxSets, \
                self.EggMethodLevels, selLevel)
        weightLst, idxSetLst, nameLst = \
                self.get_weight_index_and_name_lists_of_derived_fragments(eggNameLst, eggIdxSetInLst, selLevel)
        levelLst = [selLevel] * len(nameLst)
        pathLst = self.get_path_list_from_index_set_and_level_lists(idxSetLst, levelLst)
        return weightLst, pathLst, nameLst
    
    #3 see get_weight_file.py
    def get_weight_index_and_name_lists_of_derived_fragments(self, eggNameLst: list, eggUnitIdxSetInLst: list,\
                                                             selLevel: str, weight=1) -> tuple:
        print_xo_log("----------------------------------------------------------------------")
        print_xo_log("-------------------- Level {0} Fragment generation ---------------------".format(selLevel))
        print_xo_log("----------------------------------------------------------------------")
        uniqueUnitIdxSetLst = deepcopy(eggUnitIdxSetInLst)
        uniqueWeightLst = [weight] * len(eggUnitIdxSetInLst)
        uniqueNameSetInLst = [set([eggName]) for eggName in eggNameLst]
        currentLayerUniqueUnitIdxSetInLst = deepcopy(eggUnitIdxSetInLst)
        currentLayerUniqueWeightLst = [weight] * len(eggUnitIdxSetInLst)
        currentLayerUniqueNameSetInLst = [set([eggName]) for eggName in eggNameLst]
        currentWeightConstant = weight * -1
        iLayer = 2
        isAllSetsVoid = False
        while not isAllSetsVoid:
            isAllSetsVoid = True
            self.print_fragment_layer_statistics_for_name_set(iLayer-1, 
                    currentLayerUniqueNameSetInLst, uniqueUnitIdxSetLst)
            nextLayerUniqueUnitIdxSetInLst, nextLayerUniqueWeightLst, nextLayerUniqueNameSetInLst = [], [], []
            for currentUnitIdxSet, currentWeight, currentNameSet in zip(currentLayerUniqueUnitIdxSetInLst, \
                                                                        currentLayerUniqueWeightLst, \
                                                                        currentLayerUniqueNameSetInLst):
                for addedEggName, addedEggUnitIdxSet in zip(eggNameLst, eggUnitIdxSetInLst):
                    if addedEggName not in currentNameSet:
                        newIdxSet = currentUnitIdxSet & addedEggUnitIdxSet
                        if newIdxSet != set():
                            newNameSet = currentNameSet | {addedEggName}
                            # print("D", len(newNameSet), newNameSet, newIdxSet, currentWeight)
                            try:
                                uniqueEggIdx = nextLayerUniqueUnitIdxSetInLst.index(newIdxSet)
                                nextLayerUniqueWeightLst[uniqueEggIdx] += currentWeight
                            except ValueError:
                                nextLayerUniqueUnitIdxSetInLst.append(newIdxSet)
                                nextLayerUniqueWeightLst.append(currentWeight)
                                nextLayerUniqueNameSetInLst.append(newNameSet)
                            isAllSetsVoid = False
            if not isAllSetsVoid:
                # print("D", iLayer, currentWeightConstant)
                nonzeroLayerUniqueUnitIdxSetInLst, nonzeroLayerUniqueWeightLst, nonzeroLayerUniqueNameSetInLst = \
                        self.get_nonzero_scaled_lists(nextLayerUniqueUnitIdxSetInLst, nextLayerUniqueWeightLst, \
                                                      nextLayerUniqueNameSetInLst, iLayer * currentWeightConstant)
                for unitIdxSet, weight, nameSet in zip(nonzeroLayerUniqueUnitIdxSetInLst, \
                                                    nonzeroLayerUniqueWeightLst, \
                                                    nonzeroLayerUniqueNameSetInLst):
                    try:
                        uniqueEggIdx = uniqueUnitIdxSetLst.index(unitIdxSet)
                        uniqueWeightLst[uniqueEggIdx] += weight
                    except ValueError:
                        uniqueUnitIdxSetLst.append(unitIdxSet)
                        uniqueWeightLst.append(weight)
                        uniqueNameSetInLst.append(nameSet)
                currentWeightConstant *= -1
                iLayer += 1
                currentLayerUniqueUnitIdxSetInLst = nonzeroLayerUniqueUnitIdxSetInLst
                currentLayerUniqueWeightLst = nonzeroLayerUniqueWeightLst
                currentLayerUniqueNameSetInLst = nonzeroLayerUniqueNameSetInLst
        print_xo_log("----------------------------------------------------------------------")
        uniqueWeightLst, uniqueUnitIdxSetLst, uniqueNameLst = \
                self.get_nonzero_and_name_lists_from_lists(uniqueWeightLst, uniqueUnitIdxSetLst, uniqueNameSetInLst)
        return uniqueWeightLst, uniqueUnitIdxSetLst, uniqueNameLst

    #4
    def get_nonzero_and_name_lists_from_lists(self, weightLst: list, unitIdxSetLst: list, nameSetInLst: list) -> tuple:
        nonzeroWeightLst, nonzeroUnitIdxSetLst, nonzeroNameLst = [], [], []
        for weight, unitIdxSet, nameSet in zip(weightLst, unitIdxSetLst, nameSetInLst):
            if weight != 0:
                nonzeroWeightLst.append(weight)
                nonzeroUnitIdxSetLst.append(unitIdxSet)
                nonzeroNameLst.append(".".join(sorted(nameSet)))
        return nonzeroWeightLst, nonzeroUnitIdxSetLst, nonzeroNameLst
        
    #4
    def print_fragment_layer_statistics_for_name_set(self, iLayer: int, currentLayerUniqueNameSetInLst: list, 
                                                     uniqueUnitIdxSetLst: list):
        print_xo_log(" Layer {0:2d} has {1:5d} type(s) of fragment(s), {2:5d} unique one(s) in total".format(\
                     iLayer, len(currentLayerUniqueNameSetInLst), len(uniqueUnitIdxSetLst)))
    
    #4
    def get_nonzero_scaled_lists(self, unitIdxSetInLst: list, weightLst: list, nameSetInLst: list, scale: int) -> tuple:
        nonzeroUnitIdxSetInLst, nonzeroWeightLst, nonzeroNameSetInLst = [], [], []
        for unitIdxSet, weight, nameSet in zip(unitIdxSetInLst, weightLst, nameSetInLst):
            if weight != 0:
                if weight % scale != 0:
                    print(" Fatal error! Layer weight = {0:>8d}, whereas scaled by layer deepth = {1:>2d}".format(\
                          weight, scale))
                nonzeroUnitIdxSetInLst.append(unitIdxSet)
                nonzeroWeightLst.append(abs(weight) // scale)
                nonzeroNameSetInLst.append(nameSet)
        return nonzeroUnitIdxSetInLst, nonzeroWeightLst, nonzeroNameSetInLst
    
    #3
    def get_weight_index_and_name_lists_of_derived_fragments_ori(self, eggNameLst: list, eggUnitIdxSetInLst: list,\
                                                                 selLevel: str, weight=1) -> tuple:
        print_xo_log("----------------------------------------------------------------------")
        print_xo_log("-------------------- Level {0} Fragment generation ---------------------".format(selLevel))
        print_xo_log("----------------------------------------------------------------------")
        fragmentWeightLst = [weight] * len(eggUnitIdxSetInLst)
        fragmentUnitIdxSetLst = deepcopy(eggUnitIdxSetInLst)
        fragmentNameLst = deepcopy(eggNameLst)
        currentLayerUnitIdxSetLst = deepcopy(eggUnitIdxSetInLst)
        currentLayerNameLst = deepcopy(eggNameLst)
        currentLayerMaxEggIdxLst = list(range(len(eggNameLst)))
        currentWeight = weight * -1
        uniqueUnitIdxSetLst = deepcopy(eggUnitIdxSetInLst)
        isAllVoidSet = False
        while not isAllVoidSet:
            isAllVoidSet = True
            nextLayerUnitIdxSetLst, nextLayerNameLst, nextLayerMaxEggIdxLst = [], [], []
            self.print_fragment_layer_statistics(currentLayerNameLst, uniqueUnitIdxSetLst)
            for currentName, currentUnitIdxSet, currentMaxEggIdx in zip(currentLayerNameLst, \
                                                                        currentLayerUnitIdxSetLst, \
                                                                        currentLayerMaxEggIdxLst):
                for iEgg, (eggName, eggUnitIdxSet) in enumerate(zip(eggNameLst, eggUnitIdxSetInLst)):
                    if iEgg > currentMaxEggIdx:
                        newIdxSet = currentUnitIdxSet & eggUnitIdxSet
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
        print_xo_log("----------------------------------------------------------------------")
        return fragmentWeightLst, fragmentUnitIdxSetLst, fragmentNameLst

    #4
    def print_fragment_layer_statistics(self, currentLayerNameLst: list, uniqueUnitIdxSetLst: list):
        iLayer =currentLayerNameLst[0].count(".") + 1
        print_xo_log(" Layer {0:2d} has {1:5d} fragment(s), {2:5d} unique one(s) in total".format(iLayer, 
                     len(currentLayerNameLst), len(uniqueUnitIdxSetLst)))

    #4
    def append_new_name_to_fragment_name_string(self, nameStr: str, vName: str) -> str:
        nameLst = nameStr.split(".") + [vName]
        return ".".join(sorted(nameLst))
    
    #3
    def get_path_list_from_index_set_and_level_lists(self, idxSetLst: list, levelLst: list) -> list:
        pathLst = []
        for idxSet, uLevel in zip(idxSetLst, levelLst):
            dirName = get_dir_name_from_index_list(self.FragmentDir1stChar, idxSet)
            pathLst.append(os.path.join(self.FragmentDir, dirName, uLevel))
        return pathLst

    #2
    def get_weight_path_and_name_lists_of_derived_fragments_with_fore_and_back_grounds(self, 
                                                                                       foreLevel: str, 
                                                                                       backLevel: str) -> tuple:
        eggNameLst, eggIdxSetInLst = get_egg_name_and_index_set_lists_at_selected_level(self.EggIdxSets, \
                self.EggMethodLevels, foreLevel)
        foreWeightLst, foreIdxSetLst, foreNameLst = \
                self.get_weight_index_and_name_lists_of_derived_fragments(eggNameLst, eggIdxSetInLst, foreLevel)
        foreLevelLst = [foreLevel] * len(foreNameLst)
        forePathLst = self.get_path_list_from_index_set_and_level_lists(foreIdxSetLst, foreLevelLst)
        backWeightLst, backIdxSetLst, backNameLst = \
                self.get_background_lists_from_foreground_lists(foreWeightLst, foreIdxSetLst, foreNameLst, backLevel)
        backLevelLst = [backLevel] * len(backNameLst)
        backPathLst = self.get_path_list_from_index_set_and_level_lists(backIdxSetLst, backLevelLst)
        return foreWeightLst + backWeightLst, forePathLst + backPathLst, foreNameLst + backNameLst
    
    #3
    def get_background_lists_from_foreground_lists(self, foreWeightLst: list, foreIdxSetLst: list, foreNameLst: list,\
                                                   backLevel: str) -> tuple:
        backWeightLst, backIdxSetLst, backNameLst = [], [], []
        for uWeight, idxSet, uName in zip(foreWeightLst, foreIdxSetLst, foreNameLst):
            newSet = idxSet & self.AllIdxSetAtLevels[backLevel]
            if newSet != set([]):
                backWeightLst.append(uWeight * -1)
                backIdxSetLst.append(newSet)
                backNameLst.append(uName + "." + backLevel)
        return backWeightLst, backIdxSetLst, backNameLst
    ##### ----- #####
    
    ##### ----- #####
    #0
    def write_fragment_force_files_from_weight_and_path_alias_files(self):
        self.ElementSpins = dict() #TODO added
        # fragmentFiles = gjfFile + globalIdxFile
        rootPath = os.getcwd()
        _, uniqueFullUnitPathLst = read_weight_file(self.WeightFile)
        _, unitPath2WorkPath = read_path_alias_file(self.PathAliasFile)
        for uFullUnitPath in uniqueFullUnitPathLst:
            #fullUnitPath = .unit/U1-3_5/H
            workPath = unitPath2WorkPath[uFullUnitPath]
            mk_serial_dir(workPath)
            os.chdir(workPath)
            level, idxLst = get_method_level_and_index_list_from_calc_path(uFullUnitPath)
            fragInFile = "fragment" + self.QmManager.InputExtentions[level]
            fragOutFile = "fragment" + self.QmManager.OutputExtentions[level]
            fragCharge, fragSpin, fragXyzArray, fragElementLst, fragBondOrders, fragXyzChargeArray, \
                    globalIdxLst = self.Capper.cap_fragment_from_unit_index_list(idxLst)
            if not os.path.isfile(self.GlobalIdxFile):
                write_global_index_file(self.GlobalIdxFile, globalIdxLst)
            if not self.QmManager.is_calculation_exit_normally_for_a_type(\
                    workPath, fragOutFile, self.ResultType):
                self.QmManager.write_fragment_input_file(fragInFile, level, self.NProc, self.Mem, \
                                                         self.Methods, "force", fragCharge, fragSpin, \
                                                         fragElementLst, fragXyzArray, fragBondOrders, \
                                                         fragXyzChargeArray, self.BasisSets, self.Ecps, \
                                                         self.GaussianExtraInputs, self.ElementSpins, globalIdxLst)
                self.warn_and_remove_fragment_out_file(rootPath, workPath, fragElementLst)
            os.chdir(rootPath)
    
    #1
    def warn_and_remove_fragment_out_file(self, rootPath: str, workPath: str, fragElementLst: list, isWarn=True):
        # currently in the workPath
        level = get_method_level_from_unit_or_work_path(workPath)
        outFile = "fragment" + self.QmManager.OutputExtentions[level]
        outPathFile = os.path.join(workPath, outFile)
        if os.path.isfile(outFile):
            if isWarn:
                os.chdir(rootPath) # find the correct xo.log
                if self.ResultType == "energy":
                    print_xo_log(" Warning! 1 {0} datum is not found in {1}".format(self.ResultType, outPathFile))
                else:
                    print_xo_log(" Warning! {0} {1} data are not found in {2}".format(\
                                len(fragElementLst), self.ResultType, outPathFile))
                print_xo_log("          Remove", outPathFile)
                os.chdir(workPath)
            os.unlink(outFile)
    ##### ----- #####
    
    ##### ----- #####
    def write_high_level_full_force_input_file(self):
        path = "."
        level = get_method_level_from_unit_or_work_path(path)
        fullInFile = "full" + self.QmManager.InputExtentions[level]
        fullOutFile = "full" + self.QmManager.OutputExtentions[level]
        if not self.QmManager.is_calculation_exit_normally_for_a_type(path, fullOutFile, self.ResultType):
            self.QmManager.write_fragment_input_file(fullInFile, level, self.NProc, self.Mem, \
                                                     self.Methods, "force", self.Charge, self.Spin, \
                                                     self.ElementLst, self.Xyz, self.BondOrders, np.array([]), 
                                                     self.BasisSets, self.Ecps, self.GaussianExtraInputs)
                                                     # use globalIdxLst = [] for orca modify_orca_scf_flipspin_and_finalms
            if os.path.isfile(fullOutFile):
                if self.ResultType == "energy":
                    print_xo_log(" Warning! 1 {0} datum is not found in {1}".format(self.ResultType, fullOutFile))
                else:
                    print_xo_log(" Warning! {0} {1} data are not found in {2}".format(\
                            self.TotAtom, self.ResultType, fullOutFile))
                print_xo_log("          Remove", fullOutFile)
                os.unlink(fullOutFile)
    ##### ----- #####


class XoCalculator(XoGeneral):
    def __init__(self, inputFile: str, subFile: str, queueName: str, nProc: str, nJob: int, jobSuffix=""):
        #gRoot = "/share/apps/gaussian/G09D01_AMDGu2021"
        #        "/share/apps/gaussian/G16B01/Legacy/g16/g16"
        XoGeneral.__init__(self)
        self.SubFile = subFile
        self.QueueName = queueName
        self.NProc = str(nProc)
        self.NJob = nJob
        self.JobSuffix = jobSuffix[0:3] if len(jobSuffix) >= 3 else jobSuffix
        self.Methods, self.ExecutableCommands, self.Executables, self.GaussianExtraInputs \
                = read_data_from_config_file(self.ConfigFile)
        self.ResultType = get_result_type_from_input_file(inputFile)
        _, _, _, elementLst, _ = read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(inputFile)
        self.TotAtom = len(elementLst)
        self.QmManager = QMManager(self.ConfigFile, inputFile, queueName, nProc=nProc)
        
    #0
    def calculate_fragment_and_full(self, isFullCalculated=True):
        self.UniqueWorkPathLst = self.get_unfinished_unique_work_path_list()
        self.IsFullCalculated = isFullCalculated
        if self.QueueName.lower() == "local":
            self.calculate_fragment_and_full_locally()
        else:
            self.calculate_fragment_and_full_by_queue()
    
    #1
    def get_unfinished_unique_work_path_list(self) -> list:
        _, unitPath2WorkPath = read_path_alias_file(self.PathAliasFile)
        if os.path.isfile(self.WeightFile):
            _, uniqueUnitPathLst = read_weight_file(self.WeightFile)
        else:
            uniqueUnitPathLst = unitPath2WorkPath.keys()
        unFinishedWorkPathLst = \
                self.check_out_files_and_return_unfinished_work_path_list(unitPath2WorkPath, uniqueUnitPathLst)
        return unFinishedWorkPathLst
    
    #2
    def check_out_files_and_return_unfinished_work_path_list(self, unitPath2WorkPath: dict, 
                                                            uniqueUnitPathLst: list) -> list:
        unFinishedWorkPathLst = []
        for uUnitPath in uniqueUnitPathLst:
            level = get_method_level_from_unit_or_work_path(uUnitPath)
            outFile =  "fragment" + self.QmManager.OutputExtentions[level]
            if not self.QmManager.is_calculation_exit_normally_for_a_type(unitPath2WorkPath[uUnitPath], \
                    os.path.join(unitPath2WorkPath[uUnitPath], outFile), self.ResultType):
                unFinishedWorkPathLst.append(unitPath2WorkPath[uUnitPath])
        return unFinishedWorkPathLst
    
    #1
    def calculate_fragment_and_full_locally(self):
        workRoot = os.getcwd()
        nPath = len(self.UniqueWorkPathLst)
        for iPath, uPath in enumerate(self.UniqueWorkPathLst):
            print_xo_log(" Fragment {0:d}/{1:d} has files in {2:s} ...".format(iPath+1, nPath, uPath))
            os.chdir(os.path.join(workRoot, uPath))
            cmdLst = self.QmManager.get_command_list_from_path(os.path.join(workRoot, uPath), "fragment")
            self.run_command_list(cmdLst)
            os.chdir(workRoot)
        if self.IsFullCalculated:
            fullOutFile = "full" + self.QmManager.OutputExtentions["H"]
            if not self.QmManager.is_calculation_exit_normally_for_a_type(".", fullOutFile, self.ResultType):
                print_xo_log(" Full 1/1 has files in {0} ...".format(workRoot))
                cmdLst = self.QmManager.get_command_list_from_path(workRoot, "full")
                self.run_command_list(cmdLst)
    
    #2
    def run_command_list(self, cmdLst: list):
        for cmd in cmdLst:
            match = re.match(r"cd (.*)", cmd)
            if match:
                uDir = os.popen("echo {0}".format(match.group(1))).read()[:-1]
                os.chdir(uDir)
            else:
                os.system(cmd)
    
    #1
    def calculate_fragment_and_full_by_queue(self):
        self.QmManager.write_sub_file_and_collect_path_file_list(self.UniqueWorkPathLst, "fragment", self.NJob)
        if self.IsFullCalculated:
            self.QmManager.write_sub_file_and_collect_path_file_list(["."], "full", 1)
        self.QmManager.submit_jobs_and_write_running_job_id_to_file()
        self.QmManager.wait_till_all_jobs_normally_exit()
    
    #0
    def calculate_fragment_from_input(self, inputFile: str, queueName: str, nJob: int):
        self.QmManager = QMManager(self.ConfigFile, inputFile, queueName, nProc=self.NProc)
        uniqueWorkPathLst = self.get_unfinished_unique_work_path_list()
        if queueName.lower() == "local":
            self.calculate_fragment_locally_from_input(uniqueWorkPathLst)
        else:
            self.calculate_fragment_by_queue_from_input(inputFile, queueName, uniqueWorkPathLst, nJob)
    
    #1
    def calculate_fragment_locally_from_input(self, uniqueWorkPathLst: list):
        workRoot = os.getcwd()
        nPath = len(uniqueWorkPathLst)
        for iPath, uPath in enumerate(uniqueWorkPathLst):
            print_xo_log(" Fragment {0:d}/{1:d} has files in {2:s} ...".format(iPath+1, nPath, uPath))
            os.chdir(os.path.join(workRoot, uPath))
            cmdLst = self.QmManager.get_command_list_from_path(os.path.join(workRoot, uPath), "fragment")
            self.run_command_list(cmdLst)
            os.chdir(workRoot)
    
    #1
    def calculate_fragment_by_queue_from_input(self, inputFile: str, queueName: str, workPathLst: list, nJob: int):
        self.QmManager = QMManager(self.ConfigFile, inputFile, queueName, nProc=self.NProc)
        self.QmManager.write_sub_file_and_collect_path_file_list(workPathLst, "fragment", nJob)
        self.QmManager.submit_jobs_and_write_running_job_id_to_file()
        self.QmManager.wait_till_all_jobs_normally_exit()

class XoCollector(XoGeneral):
    def __init__(self, resultType=None, gjfFile=None):
        # input properties
        XoGeneral.__init__(self)
        # derived properties
        _, self.Spin, _, self.ElementLst, _ = \
                read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(gjfFile)
        self.NFullSpin = (self.Spin - 1) / 2 #todo: can be read from ORCA (when flipped)
        if resultType is None:
            self.ResultType = get_result_type_from_input_file(gjfFile)
        else:
            self.ResultType = resultType
        self.QmManager = QMManager(self.ConfigFile, gjfFile)
        self.FragmentVars, self.GlobalAtomIdxLsts, self.LocalAtomIdxLsts = dict(), dict(), dict()
    
    ##### ----- #####
    #0
    def collect_xo_result(self):
        self.WeightLst, self.UniqueUnitPathLst = read_weight_file(self.WeightFile)
        _, self.UnitPath2WorkPath = read_path_alias_file(self.PathAliasFile)
        self.UniqueWorkPathLst = [self.UnitPath2WorkPath[uPath] for uPath in self.UniqueUnitPathLst]
        if self.ResultType == "force":
            self.XoResult = self.collect_xo_force()
        elif self.ResultType == "energy":
            self.XoResult = self.collect_xo_energy()
        elif self.ResultType == "hessian":
            self.XoResult = self.collect_xo_hessian()
        elif self.ResultType == "nmr":
            self.XoResult = self.collect_xo_nmr()
        elif self.ResultType == "tensor":
            self.XoResult = self.collect_xo_tensor_dict() # xoTensors["G"], ["A"], ["X"]
        else:
            print_error_and_exit("XO result type =", self.ResultType, "is invalid.")
        return self.XoResult
    
    #0
    def collect_xo_result_list(self):
        self.WeightLst, self.UniqueUnitPathLst = read_weight_file(self.WeightFile)
        _, self.UnitPath2WorkPath = read_path_alias_file(self.PathAliasFile)
        self.UniqueWorkPathLst = [self.UnitPath2WorkPath[uPath] for uPath in self.UniqueUnitPathLst]
        if self.ResultType == "force":
            xoEnergy = self.collect_xo_energy()
            self.clear_fragment_vars()
            xoForce = self.collect_xo_force()
            self.XoResultLst = [xoEnergy, xoForce]
        elif self.ResultType == "energy":
            self.XoResultLst = [self.collect_xo_energy()]
        # elif self.ResultType == "nmr":
        #     self.XoResult = self.collect_xo_nmr()
        # elif self.ResultType == "tensor":
        #     self.XoResult = self.collect_xo_tensor()
        elif self.ResultType == "hessian":
            xoEnergy = self.collect_xo_energy()
            self.clear_fragment_vars()
            xoForce = self.collect_xo_force()
            self.clear_fragment_vars()
            xoHessian = self.collect_xo_hessian()
            self.XoResultLst = [xoEnergy, xoForce, xoHessian]
        else:
            print_error_and_exit("XO result type =", self.ResultType, "is invalid.")
        return self.XoResultLst
    
    #1
    def clear_fragment_vars(self):
        self.FragmentVars = dict()
        # self.FragmentVars, self.GlobalAtomIdxLsts, self.LocalAtomIdxLsts = dict(), dict(), dict()
    #1
    def collect_xo_hessian(self) -> np.ndarray:
        MAXATOM = 10000
        maxAtom = -1
        xoHessian = np.zeros((MAXATOM*3, MAXATOM*3))
        [hessians, _, _], globalAtomIdxLsts, localAtomIdxLsts = \
                self.read_var_and_index_dicts(self.QmManager.read_hessian_function_dict())
        for weight, uPath in zip(self.WeightLst, self.UniqueUnitPathLst):
            if np.max(globalAtomIdxLsts[uPath]) > maxAtom:
                maxAtom = np.max(globalAtomIdxLsts[uPath])
                if maxAtom > MAXATOM - 1:
                    print_error_and_exit("Too many atoms in the system. Please increase the value of MAXATOM in the \
                                          collect_xo_hessian subroutine.")
            iLocalLst, jLocalLst = [], []
            iGlobalLst, jGlobalLst = [], []
            # print("DP", uPath)
            # print("D1", localAtomIdxLsts[uPath])
            # print("D2", globalAtomIdxLsts[uPath])
            for iLocal, iGlobal in zip(localAtomIdxLsts[uPath], globalAtomIdxLsts[uPath]):
                for jLocal, jGlobal in zip(localAtomIdxLsts[uPath], globalAtomIdxLsts[uPath]):
                    for i in range(3):
                        for j in range(3):
                            iLocalLst.extend([iLocal*3 + i])
                            jLocalLst.extend([jLocal*3 + j])
                            iGlobalLst.extend([iGlobal*3 + i])
                            jGlobalLst.extend([jGlobal*3 + j])
            # print("DW", weight)
            # print("DI", iLocalLst)
            # print("DJ", jLocalLst)
            # print("DD", hessians[uPath][iLocalLst, jLocalLst].shape)
            # print("DX", ["{:10.6f}".format(u) for u in xoHessian[iGlobalLst, jGlobalLst]])
            # print("DH", ["{:10.6f}".format(u) for u in hessians[uPath][iLocalLst, jLocalLst]])
            # for uLst in hessians[uPath]:
            #     for u in uLst:
            #         print("{:10.6f}".format(u), end=" ")
            #     print()
            # print("X------------------------X")
            # for uLst in xoHessian:
            #     for u in uLst:
            #         print("{:10.6f}".format(u), end=" ")
            #     print()
            # print("B------------------------B")
            xoHessian[iGlobalLst, jGlobalLst] += hessians[uPath][iLocalLst, jLocalLst] * weight
            # for uLst in xoHessian:
            #     for u in uLst:
            #         print("{:10.6f}".format(u), end=" ")
            #     print()
            # print("A------------------------A")
        return xoHessian[0:maxAtom*3+3, 0:maxAtom*3+3]
    
    #1
    def collect_xo_force(self) -> np.ndarray:
        MAXATOM = 50001
        maxAtom = -1
        xoForce = np.zeros((MAXATOM, 3))
        [forces, _, _], globalAtomIdxLsts, localAtomIdxLsts = \
                self.read_var_and_index_dicts(self.QmManager.read_force_function_dict())
        for weight, uPath in zip(self.WeightLst, self.UniqueUnitPathLst):
            if np.max(globalAtomIdxLsts[uPath]) > maxAtom:
                maxAtom = np.max(globalAtomIdxLsts[uPath])
                if maxAtom > MAXATOM - 1:
                    print_error_and_exit("Too many atoms in the system. Please increase the value of MAXATOM in the \
                                          collect_xo_force subroutine.")
            xoForce[globalAtomIdxLsts[uPath], :] += forces[uPath][localAtomIdxLsts[uPath], :] * weight
        return xoForce[0:maxAtom+1, :]
    
    #2
    def read_var_and_index_dicts(self, read_var_functions: dict) -> tuple:
        self.FragmentVarsLst = [dict(), dict(), dict()]
        for uUnitPath, uWorkPath in zip(self.UniqueUnitPathLst, self.UniqueWorkPathLst):
            level = get_method_level_from_unit_or_work_path(uWorkPath)
            outFile = "fragment" + self.QmManager.OutputExtentions[level]
            if uUnitPath not in self.FragmentVarsLst[0].keys() or len(self.FragmentVarsLst[0][uUnitPath]) == 0:
                if type(read_var_functions[level]) != type([]):
                    self.FragmentVarsLst[0][uUnitPath] = read_var_functions[level](os.path.join(uWorkPath, outFile))
                else:
                    # when read_var_functions[level] = [read_orca_g_tensor, read_orca_a_tensor, read_orca_superexchange]
                    for i, uFunction in enumerate(read_var_functions[level]):
                        self.FragmentVarsLst[i][uUnitPath] = uFunction(os.path.join(uWorkPath, outFile))
                if len(self.FragmentVarsLst[0][uUnitPath]) == 0:
                    print_error_and_exit("Unable to read valid data from {0}".format(\
                                         os.path.join(uWorkPath, outFile)))
            if uUnitPath not in self.GlobalAtomIdxLsts.keys() or uUnitPath not in self.LocalAtomIdxLsts.keys():
                self.GlobalAtomIdxLsts[uUnitPath], self.LocalAtomIdxLsts[uUnitPath] = \
                        read_global_index_file(os.path.join(uWorkPath, self.GlobalIdxFile))
        return self.FragmentVarsLst, self.GlobalAtomIdxLsts, self.LocalAtomIdxLsts
    
    #1
    def collect_xo_energy(self) -> float:
        xoEnergy = 0.0
        [energies, _, _], _, _ = self.read_var_and_index_dicts(self.QmManager.read_energy_function_dict())
        for weight, uPath in zip(self.WeightLst, self.UniqueUnitPathLst):
            xoEnergy += energies[uPath] * weight
        return xoEnergy

    #1
    def collect_xo_nmr(self) -> np.ndarray:
        MAXATOM = 50001
        maxAtom = -1
        xoChemShift = np.zeros(MAXATOM)
        [chemShifts, _, _], globalAtomIdxLsts, localAtomIdxLsts \
                = self.read_var_and_index_dicts(self.QmManager.read_nmr_function_dict())
        for weight, uPath in zip(self.WeightLst, self.UniqueUnitPathLst):
            if np.max(globalAtomIdxLsts[uPath]) > maxAtom:
                maxAtom = np.max(globalAtomIdxLsts[uPath])
                if maxAtom > MAXATOM - 1:
                    print_error_and_exit("Too many atoms in the system. Please increase the value of MAXATOM in the \
                                          collect_xo_nmr subroutine.")
            xoChemShift[globalAtomIdxLsts[uPath]] += chemShifts[uPath][localAtomIdxLsts[uPath]] * weight
        return xoChemShift[0:maxAtom+1]
    
    #1
    def collect_xo_tensor_dict(self) -> dict:
        MAXATOM = 50001
        maxAtom = -1
        xoTensors = dict()
        xoTensors["G"] = np.zeros((3, 3))
        xoTensors["A"] = np.zeros((MAXATOM, 3, 3))
        xoTensors["X"] = 0.0
        [gTensors, scaledATensor3dArrays, superexchanges], globalAtomIdxLsts, localAtomIdxLsts \
                = self.read_var_and_index_dicts(self.QmManager.read_tensor_dict())
        for weight, uPath in zip(self.WeightLst, self.UniqueUnitPathLst):
            if np.max(globalAtomIdxLsts[uPath]) > maxAtom:
                maxAtom = np.max(globalAtomIdxLsts[uPath])
                if maxAtom > MAXATOM - 1:
                    print_error_and_exit("Too many atoms in the system. Please increase the value of MAXATOM in the \
                                          collect_xo_tensor_dict subroutine.")
            xoTensors["G"] += gTensors[uPath] * weight
            xoTensors["A"][globalAtomIdxLsts[uPath]] += \
                    scaledATensor3dArrays[uPath][localAtomIdxLsts[uPath]] * weight / self.NFullSpin
            # print("D", uPath, self.NFullSpin, scaledATensor3dArrays[uPath][localAtomIdxLsts[uPath]])
            xoTensors["X"] += superexchanges[uPath] * weight
        return xoTensors
    ##### ----- #####
    
    ##### ----- #####
    def write_xo_result_to_file(self, suffix=""):
        if self.ResultType == "force":
            self.write_force_file("xo_" + self.ResultType + suffix, self.XoResult)
        elif self.ResultType == "energy":
            self.write_energy_file("xo_" + self.ResultType + suffix, self.XoResult)
        elif self.ResultType == "hessian":
            self.write_hessian_file("xo_" + self.ResultType + suffix, self.XoResult)
        elif self.ResultType == "nmr":
            self.write_nmr_file("xo_" + self.ResultType + suffix, self.XoResult, self.ElementLst)
        elif self.ResultType == "tensor":
            self.write_g_tensor_file("xo_g_" + self.ResultType + suffix, self.XoResult["G"])
            self.write_a_tensor_file("xo_a_" + self.ResultType + suffix, self.XoResult["A"], self.ElementLst)
            self.write_energy_file("xo_superexchange" + suffix, [self.XoResult["X"]])
        else:
            print_error_and_exit("XO result type =", self.ResultType, "is invalid.")
    
    ###
    def write_force_file(self, fileName: str, force: np.ndarray):
        backup_serial_file(fileName)
        with open(fileName, "w") as wfl:
            for f in force:
                print("{0:15.9f} {1:15.9f} {2:15.9f}".format(f[0], f[1], f[2]), file=wfl)
    
    ###
    def write_energy_file(self, fileName: str, energy: np.ndarray):
        backup_serial_file(fileName)
        with open(fileName, "w") as wfl:
            print("{0:26.16f}".format(energy[0]), file=wfl)
    
    ###
    def write_hessian_file(self, fileName: str, hessian: np.ndarray):
        backup_serial_file(fileName)
        with open(fileName, "w") as wfl:
            for i, hessArray in enumerate(hessian):
                for j, uhess in enumerate(hessArray):
                    if j <= i:
                        print("{:15.8f} ".format(uhess), end="", file=wfl)
                print(file=wfl)

    ###
    def write_nmr_file(self, fileName: str, chemShift: np.ndarray, elementLst=[]):
        backup_serial_file(fileName)
        if elementLst == []:
            with open(fileName, "w") as wfl:
                for cs in chemShift:
                    print("{0:15.9f}".format(cs), file=wfl)
        else:
            with open(fileName, "w") as wfl:
                for iAtom, (element, cs) in enumerate(zip(elementLst, chemShift)):
                    print("{0:5d} {1:2s} {2:15.9f}".format(iAtom+1, element, cs), file=wfl)
    
    #1
    def write_g_tensor_file(self, fileName: str, tensorArray: np.ndarray):
        backup_serial_file(fileName)
        with open(fileName, "w") as wfl:
            for row in tensorArray:
                print("{0:10.6f} {1:10.6f} {2:10.6f}".format(row[0], row[1], row[2]), file=wfl)
    
    #1
    def write_a_tensor_file(self, fileName: str, tensor3dArray: np.ndarray, elementLst: list):
        backup_serial_file(fileName)
        with open(fileName, "w") as wfl:
            for iAtom, (element, tensor) in enumerate(zip(elementLst, tensor3dArray)):
                print("{0}{1}".format(iAtom, element), file=wfl)
                for row in tensor:
                    print("{0:10.6f} {1:10.6f} {2:10.6f}".format(row[0], row[1], row[2]), file=wfl)
    ##### ----- #####
    
    ##### ----- #####
    #0
    def read_full_result(self):
        fullOutFile = "full" + self.QmManager.OutputExtentions["H"]
        if self.ResultType == "force":
            self.FullResult = self.QmManager.read_force_function_dict()["H"](fullOutFile)
        elif self.ResultType == "energy":
            self.FullResult = self.QmManager.read_energy_function_dict()["H"](fullOutFile)
        elif self.ResultType == "hessian":
            self.FullResult = self.QmManager.read_hessian_function_dict()["H"](fullOutFile)
        elif self.ResultType == "nmr":
            self.FullResult = self.QmManager.read_nmr_function_dict()["H"](fullOutFile)
        else:
            print_error_and_exit("XO result type = {0} is invalid.".format(self.ResultType))
    
    #0
    def collect_full_result(self) -> np.ndarray:
        self.read_full_result()
        return self.FullResult
    ##### ----- #####
    
    ##### ----- #####
    #0
    def write_full_result_to_file(self):
        if self.ResultType == "force":
            self.write_force_file("full_" + self.ResultType, self.FullResult)
        elif self.ResultType == "energy":
            self.write_energy_file("full_" + self.ResultType, self.FullResult)
        elif self.ResultType == "hessian":
            self.write_energy_file("full_" + self.ResultType, self.FullResult)
        elif self.ResultType == "nmr":
            self.write_nmr_file("full_" + self.ResultType, self.FullResult)
        else:
            print_error_and_exit("XO result type = {0} is invalid.".format(self.ResultType))
    ##### ----- #####

    ##### ----- #####
    #0
    def calculate_difference(self):
        if self.ResultType == "force":
            return np.sqrt(np.sum((self.XoResult - self.FullResult)**2, axis=1))
        elif self.ResultType == "energy":
            return self.XoResult - self.FullResult
        elif self.ResultType == "hessian":
            return self.XoResult - self.FullResult
        elif self.ResultType == "nmr":
            return self.XoResult - self.FullResult
        else:
            print_error_and_exit("XO result type = {0} is invalid.".format(self.ResultType))
    ##### ----- #####

class UnitFragmentOptimizer(XoSetup):
    def __init__(self, gjfFile: str, unitFile: str, connectivityFile: str, subFile: str, fragmentDir: str, \
                 queueName: str, nProc: str, nJob: int, updateMode="UXNM", \
                 neighborType="C", connectivityCutoff=0.002, nOptRound=1, xoErrorThreshold=0.0005, \
                 initEggLevel=0, initMergeCenterLevel=-1, initMergeNeighborLevel=-1, \
                 initRadius=3.0, incrRadius=2.0, deltaRadius=0.1, maxIncrHvyAtom=100, nEggLevel=3, \
                 radiusLst=[], backChargeRange=0.0, backChargeFile=""):
        XoGeneral.__init__(self)
        self.QmManager = QMManager(self.ConfigFile, gjfFile, queueName, nProc=nProc)
        self.get_standard_orientation(gjfFile, self.GjfFile, self.QmManager)
        # input properties
        self.UnitFile = unitFile
        self.ConnectivityFile = connectivityFile
        self.SubFile = subFile
        self.FragmentDir = fragmentDir
        self.QueueName = queueName
        self.NProc = str(nProc)
        self.NJob = nJob
        self.UpdateMode = self.get_valid_update_mode(updateMode)
        self.NeighborType = self.get_valid_neighbor_type(neighborType)
        self.ConnectivityCutoff = connectivityCutoff
        self.NOptRound = nOptRound
        self.XoErrorThreshold = xoErrorThreshold
        self.InitEggLevel = initEggLevel
        self.InitMergeCenterLevel = initMergeCenterLevel
        self.InitMergeNeighborLevel = initMergeNeighborLevel
        self.InitRadius = initRadius
        self.IncrRadius = incrRadius
        self.DeltaRadius = deltaRadius
        self.MaxIncrHvyAtom = maxIncrHvyAtom
        self.NEggLevel = nEggLevel
        self.MaxEggLevel = nEggLevel - 1
        self.RadiusLst = radiusLst
        self.BackChargeRange = backChargeRange
        self.BackChargeFile = backChargeFile
        # derived properties
        self.U = self.get_unit_object()
    
    ##### ----- #####
    def get_valid_update_mode(self, updateMode: str) -> str:
        if updateMode.upper() in self.ValidUpdateModeLst:
            return updateMode.upper()
        else:
            print_error_and_exit("Invalid update mode =", updateMode, "is provided.")
    ##### ----- #####

    ##### ----- #####
    #####
    def optimize(self):
        iOptRound = 1
        maxError = float("inf")
        while iOptRound <= self.NOptRound and maxError > self.XoErrorThreshold:
            if iOptRound == 1:
                self.EggLevelArray = self.get_initial_egg_level_array(self.InitEggLevel)
                self.MergeIdxLsts = self.initialize_unit_index_dict()
                self.MergeCenterLevels = self.initialize_unit_level_dict(self.InitMergeCenterLevel)
                self.MergeNeighborLevels = self.initialize_unit_level_dict(self.InitMergeNeighborLevel)
            else:
                self.update_egg(errorArray) # -> new self.EggLevelArray, self.MergeIdxLsts
            errorArray, maxError, highEggIdxSetLst = self.calculate_error()
            self.print_optimization_round_information(iOptRound, errorArray, highEggIdxSetLst)
            iOptRound += 1
    #####

    ####
    def get_initial_egg_level_array(self, intVal: int) -> np.ndarray:
        return np.ones(self.U.N).astype("int") * intVal

    ####
    def initialize_unit_index_dict(self) -> dict:
        newDict = dict()
        for i in range(self.U.N):
            newDict[i] = []
        return newDict

    ####
    def initialize_unit_level_dict(self, val=0) -> dict:
        newDict = dict()
        for i in range(self.U.N):
            newDict[i] = val
        return newDict

    ####
    def calculate_error(self) -> tuple:
        return self.calculate_force_error()
        """
        if self.ErrorType == "F":
            return self.calculate_force_error()
        elif self.ErrorType == "C":
            return self.calculate_chemical_shift_error()
        else:
            print_error_and_exit("Invalid error type =", self.ErrorType)
        """

    ###
    def calculate_force_error(self):
        xoSetup = XoSetup(self.GjfFile, self.UnitFile, self.ConnectivityFile, self.FragmentDir, self.NProc, \
                          self.EggLevelArray, self.NeighborType, self.ConnectivityCutoff, \
                          self.InitRadius, self.IncrRadius, self.DeltaRadius, self.MaxIncrHvyAtom, self.NEggLevel, \
                          self.RadiusLst, self.BackChargeRange, self.BackChargeFile)
        print_xo_log(" Generating eggs and write to egg file ...")
        xoSetup.write_xo2_full_egg_file_from_egg_level_array_and_merge_dict(self.EggLevelArray, self.MergeIdxLsts)
        print_xo_log(" Generating fragments and write to weight file...")
        xoSetup.get_fragment_weight_and_dir_lists_and_write_to_weight_and_path_alias_files()
        print_xo_log(" Writing input calculation files ...")
        xoSetup.write_fragment_force_files_from_weight_and_path_alias_files()
        xoSetup.write_high_level_full_force_input_file()
        print_xo_log(" Submitting jobs to queue for calculation ...")
        xoCalculator = XoCalculator(self.GjfFile, self.SubFile, self.QueueName, self.NProc, self.NJob, self.GjfFile)
        xoCalculator.calculate_fragment_and_full()
        print_xo_log(" Collecting results ...")
        xoCollector = XoCollector(resultType="force", gjfFile=self.GjfFile)
        self.XoResult = xoCollector.collect_xo_result()
        xoCollector.read_full_result()
        print_xo_log(" Writing results to files ...")
        xoCollector.write_xo_result_to_file()
        xoCollector.write_full_result_to_file()
        forceErrorArray = xoCollector.calculate_difference()
        _, highEggIdxSetLst = get_egg_name_and_index_set_lists_at_selected_level(xoSetup.EggIdxSets, \
                xoSetup.EggMethodLevels, "H")
        return forceErrorArray, np.max(forceErrorArray), highEggIdxSetLst
    
    ###
    def calculate_chemical_shift_error(self):
        pass #todo

    ####
    def update_egg(self, errorArray: np.ndarray):
        if self.UpdateMode == "LXM" or self.UpdateMode == "LEVELEXPANDMERGE": #expand + merge
            self.update_egg_with_level_expand_plus_merge(errorArray)
        elif self.UpdateMode == "UXCM" or self.UpdateMode == "UNITEXPANDCENTERMERGE": #unit expand + center merge
            self.update_egg_by_expand_plus_center_merge(errorArray, "UXCM")
        elif self.UpdateMode == "LXCM" or self.UpdateMode == "LEVELEXPANDCENTERMERGE": #level expand + center merge
            self.update_egg_by_expand_plus_center_merge(errorArray, "LXCM")
        elif self.UpdateMode == "UXNM" or self.UpdateMode == "UNITEXPANDNEIGHBORMERGE": #unit expand + neighbor merge
            self.update_egg_by_expand_plus_neighbor_merge(errorArray, "UXNM")
        elif self.UpdateMode == "LXNM" or self.UpdateMode == "LEVELEXPANDNEIGHBORMERGE": #level expand + neighbor merge
            self.update_egg_by_expand_plus_neighbor_merge(errorArray, "LXNM")
        elif self.UpdateMode == "SXM" or self.UpdateMode == "SIZEEXPANDMERGE": #expand + merge
            self.update_egg_with_size_wise_expand_plus_merge(errorArray)
        elif self.UpdateMode == "UX" or self.UpdateMode == "UNITEXPAND":
            self.update_egg_by_unit_expand(errorArray)
        elif self.UpdateMode == "LX" or self.UpdateMode == "LEVELEXPAND":
            self.update_egg_by_level_expand(errorArray)
        else:
            print_error_and_exit("Unsupported update mode =", self.UpdateMode)
    
    ###
    def update_egg_by_expand_plus_neighbor_merge(self, errorArray: np.ndarray, updateMode: str):
        newMergeCenterLevels, newMergeNeighborLevels, newMergeIdxLsts = \
                self.get_new_merge_levels_and_index_list_dicts_if_neighbor_merge_and_return_false_when_fail(errorArray)
        if updateMode == "LXNM":
            newEggLevelArray, errorAtomIdx, errorUnitIdx = \
                    self.get_new_egg_level_array_if_level_wise_expand_and_return_false_when_fail(errorArray)
        elif updateMode == "UXNM":
            newEggLevelArray, errorAtomIdx, errorUnitIdx = \
                    self.get_new_egg_level_array_if_unit_wise_expand_and_return_false_when_fail(errorArray)
        else:
            print(" Fatal error! Invalid update mode =", updateMode)
            sys.exit(1)
        #
        if type(newMergeCenterLevels) == type(False) and type(newEggLevelArray) == type(False):
            print_xo_log(" Warning! No", updateMode, "level update can be done for any atom.")
            sys.exit(0)
        elif type(newMergeCenterLevels) == type(False) and type(newEggLevelArray) != type(False):
            self.EggLevelArray = newEggLevelArray
            print_xo_log(" {0} Level eXpand mode used for unit {1} atom {2} in the next optimization round.".format(\
                    updateMode, errorUnitIdx+1, errorAtomIdx+1))
        elif type(newMergeCenterLevels) != type(False) and type(newEggLevelArray) == type(False):
            self.MergeCenterLevels = newMergeCenterLevels
            self.MergeNeighborLevels = newMergeNeighborLevels
            self.MergeIdxLsts = newMergeIdxLsts
            print_xo_log(" {0} Level Merge mode used for unit {1} atom {2} in the next optimization round.".format(\
                    updateMode, errorUnitIdx, errorAtomIdx))
        else:
            maxHvAtmInEggAfterMerge = self.get_max_egg_size(self.EggLevelArray, newMergeIdxLsts)
            maxHvAtmInEggAfterExpand = self.get_max_egg_size(newEggLevelArray, self.MergeIdxLsts)
            print_xo_log(" {0} Level estimated heavy atom numbers after merging ({1}) vs after expanding ({2})"\
                    .format(updateMode, maxHvAtmInEggAfterMerge, maxHvAtmInEggAfterExpand))
            if maxHvAtmInEggAfterMerge < maxHvAtmInEggAfterExpand:
                self.MergeCenterLevels = newMergeCenterLevels
                self.MergeNeighborLevels = newMergeNeighborLevels
                self.MergeIdxLsts = newMergeIdxLsts
                print_xo_log(" {0} Level Merge mode selected for unit {1} atom {2} in the next optimization round."\
                        .format(updateMode, errorUnitIdx, errorAtomIdx))
            else:
                self.EggLevelArray = newEggLevelArray
                print_xo_log(" {0} Level eXpand mode selected for unit {1} atom {2} in the next optimization round."\
                        .format(updateMode, errorUnitIdx, errorAtomIdx))

    ##
    def get_new_merge_levels_and_index_list_dicts_if_neighbor_merge_and_return_false_when_fail(self, \
            errorArray: np.ndarray) -> tuple:
        newMergeCenterLevels = deepcopy(self.MergeCenterLevels)
        newMergeNeighborLevels = deepcopy(self.MergeNeighborLevels)
        for centerAtomIdx in np.argsort(errorArray)[::-1]:
            centerUnitIdx = self.U.get_unit_index_from_atom_index(centerAtomIdx)
            # print("DU", centerUnitIdx, errorArray[centerAtomIdx], "L", iLevel)
            if newMergeCenterLevels[centerUnitIdx] < self.EggLevelArray[centerUnitIdx] and \
                    newMergeNeighborLevels[centerUnitIdx] == newMergeCenterLevels[centerUnitIdx]:
                newMergeCenterLevels[centerUnitIdx] += 1
                newMergeIdxLsts = self.get_merge_index_list_dict_from_merge_level_dicts(newMergeCenterLevels, \
                                                                                        newMergeNeighborLevels)
                return newMergeCenterLevels, newMergeNeighborLevels, newMergeIdxLsts
            elif newMergeCenterLevels[centerUnitIdx] == self.EggLevelArray[centerUnitIdx] and \
                    newMergeNeighborLevels[centerUnitIdx] < newMergeCenterLevels[centerUnitIdx]:
                newMergeNeighborLevels[centerUnitIdx] += 1
                newMergeIdxLsts = self.get_merge_index_list_dict_from_merge_level_dicts(newMergeCenterLevels, \
                                                                                        newMergeNeighborLevels)
                return newMergeCenterLevels, newMergeNeighborLevels, newMergeIdxLsts
            elif newMergeCenterLevels[centerUnitIdx] < self.EggLevelArray[centerUnitIdx] and \
                    newMergeNeighborLevels[centerUnitIdx] < newMergeCenterLevels[centerUnitIdx]:
                maxHvAtmInEggAfterCenterMerge = self.get_max_egg_size_after_center_merge(newMergeCenterLevels, 
                        newMergeNeighborLevels, centerUnitIdx)
                maxHvAtmInEggAfterNeighborMerge = self.get_max_egg_size_after_neighbor_merge(newMergeCenterLevels, 
                        newMergeNeighborLevels, centerUnitIdx)
                print_xo_log(" Estimated heavy atom numbers after center merging ({0}) vs after neighbor merging ({1})"\
                        .format(maxHvAtmInEggAfterCenterMerge, maxHvAtmInEggAfterNeighborMerge))
                if maxHvAtmInEggAfterCenterMerge <= maxHvAtmInEggAfterNeighborMerge:
                    newMergeCenterLevels[centerUnitIdx] += 1
                else:
                    newMergeNeighborLevels[centerUnitIdx] += 1
                newMergeIdxLsts = self.get_merge_index_list_dict_from_merge_level_dicts(newMergeCenterLevels, \
                                                                                        newMergeNeighborLevels)
                return newMergeCenterLevels, newMergeNeighborLevels, newMergeIdxLsts
        return False, False, False

    def get_max_egg_size_after_center_merge(self, mergeCenterLevels: dict, mergeNeighborLevels: dict, 
                                            unitIdx: int) -> int:
        mergeCenterLevels[unitIdx] += 1
        mergeIdxLstsAfterCenterMerge = self.get_merge_index_list_dict_from_merge_level_dicts(\
                mergeCenterLevels, mergeNeighborLevels)
        mergeCenterLevels[unitIdx] -= 1
        return self.get_max_egg_size(self.EggLevelArray, mergeIdxLstsAfterCenterMerge)

    def get_max_egg_size_after_neighbor_merge(self, mergeCenterLevels: dict, mergeNeighborLevels: dict, 
                                              unitIdx: int) -> int:
        mergeNeighborLevels[unitIdx] += 1
        mergeIdxLstsAfterNeighborMerge = self.get_merge_index_list_dict_from_merge_level_dicts(\
                mergeCenterLevels, mergeNeighborLevels)
        mergeNeighborLevels[unitIdx] -= 1
        return self.get_max_egg_size(self.EggLevelArray, mergeIdxLstsAfterNeighborMerge)
    
    #m
    def get_merge_index_list_dict_from_merge_level_dicts(self, mergeCenterLevels: dict, 
                                                         mergeNeighborLevels: dict) -> dict:
        newMergeIdxLsts = dict()
        for iUnitIdx in mergeCenterLevels.keys():
            if mergeCenterLevels[iUnitIdx] < 0:
                newMergeIdxLsts[iUnitIdx] = []
            else:
                newMergeIdxLsts[iUnitIdx] = \
                        list(self.U.UnitNeighborIdxSetLstsAsLevelLst[mergeCenterLevels[iUnitIdx]][iUnitIdx][0])
        for iUnitIdx in mergeNeighborLevels.keys():
            if mergeNeighborLevels[iUnitIdx] >= 0:
                unitLst = list(self.U.UnitNeighborIdxSetLstsAsLevelLst[mergeNeighborLevels[iUnitIdx]][iUnitIdx][0])
                for jUnitIdx in unitLst:
                    newMergeIdxLsts[jUnitIdx] += unitLst
        for iUnitIdx in newMergeIdxLsts.keys():
            newMergeIdxLsts[iUnitIdx] = list(set(newMergeIdxLsts[iUnitIdx]))
        return newMergeIdxLsts
    
    ###
    def update_egg_by_expand_plus_center_merge(self, errorArray: np.ndarray, updateMode: str):
        newMergeCenterLevels, newMergeIdxLsts = \
                self.get_new_merge_center_level_and_index_list_dicts_if_merge_and_return_false_when_fail(errorArray)
        if updateMode == "LXCM":
            newEggLevelArray, errorAtomIdx, errorUnitIdx = \
                    self.get_new_egg_level_array_if_level_wise_expand_and_return_false_when_fail(errorArray)
        elif updateMode == "UXCM":
            newEggLevelArray, errorAtomIdx, errorUnitIdx = \
                    self.get_new_egg_level_array_if_unit_wise_expand_and_return_false_when_fail(errorArray)
        else:
            print(" Fatal error! Invalid update mode =", updateMode)
            sys.exit(1)
        if type(newMergeCenterLevels) == type(False) and type(newEggLevelArray) == type(False):
            print_xo_log(" Warning! No", updateMode, "level update can be done for any atom.")
            sys.exit(0)
        elif type(newMergeCenterLevels) == type(False) and type(newEggLevelArray) != type(False):
            self.EggLevelArray = newEggLevelArray
            print_xo_log(" {0} Level eXpand mode used for unit {1} atom {2} in the next optimization round.".format(\
                    updateMode, errorUnitIdx, errorAtomIdx))
        elif type(newMergeCenterLevels) != type(False) and type(newEggLevelArray) == type(False):
            self.MergeCenterLevels = newMergeCenterLevels
            self.MergeIdxLsts = newMergeIdxLsts
            print_xo_log(" {0} Level Merge mode used for unit {1} atom {2} in the next optimization round.".format(\
                    updateMode, errorUnitIdx, errorAtomIdx))
        else:
            maxHvAtmInEggAfterMerge = self.get_max_egg_size(self.EggLevelArray, newMergeIdxLsts)
            maxHvAtmInEggAfterExpand = self.get_max_egg_size(newEggLevelArray, self.MergeIdxLsts)
            if maxHvAtmInEggAfterMerge < maxHvAtmInEggAfterExpand:
                self.MergeCenterLevels = newMergeCenterLevels
                self.MergeIdxLsts = newMergeIdxLsts
                print_xo_log(" {0} Level Merge mode selected for unit {1} atom {2} in the next optimization round.".format(\
                        updateMode, errorUnitIdx, errorAtomIdx))
            else:
                self.EggLevelArray = newEggLevelArray
                print_xo_log(" {0} Level eXpand mode selected for unit {1} atom {2} in the next optimization round.".format(\
                        updateMode, errorUnitIdx, errorAtomIdx))

    ##
    def get_new_merge_center_level_and_index_list_dicts_if_merge_and_return_false_when_fail(self, \
            errorArray: np.ndarray) -> tuple:
        newMergeIdxLsts = dict()
        newMergeCenterLevels = deepcopy(self.MergeCenterLevels)
        for centerAtomIdx in np.argsort(errorArray)[::-1]:
            centerUnitIdx = self.U.get_unit_index_from_atom_index(centerAtomIdx)
            iEggLevel = self.EggLevelArray[centerUnitIdx]
            # print("DU", centerUnitIdx, errorArray[centerAtomIdx], "L", iLevel)
            if newMergeCenterLevels[centerUnitIdx] < iEggLevel:
                newMergeCenterLevels[centerUnitIdx] += 1
                newMergeIdxLsts = self.get_merge_index_list_dict_from_egg_level_dict(newMergeCenterLevels)
                return newMergeCenterLevels, newMergeIdxLsts
        return False, False

    #
    def get_merge_index_list_dict_from_egg_level_dict(self, mergeLevels: dict) -> dict:
        newMergeIdxLsts = dict()
        for iUnitIdx in mergeLevels.keys():
            newMergeIdxLsts[iUnitIdx] = \
                    list(self.U.UnitNeighborIdxSetLstsAsLevelLst[mergeLevels[iUnitIdx]][iUnitIdx][0])
        return newMergeIdxLsts

    ##
    def get_new_egg_level_array_if_level_wise_expand_and_return_false_when_fail(self, \
            errorArray: np.ndarray) -> np.ndarray:
        newEggLevelArray = deepcopy(self.EggLevelArray)
        minLevel, maxLevel = np.min(newEggLevelArray), np.max(newEggLevelArray)
        if maxLevel == minLevel:
            for errorAtomIdx in np.argsort(errorArray)[::-1]:
                errorUnitIdx = self.U.get_unit_index_from_atom_index(errorAtomIdx)
            if newEggLevelArray[errorUnitIdx] < self.MaxEggLevel:
                newEggLevelArray[errorUnitIdx] += 1
                return newEggLevelArray, errorAtomIdx, errorUnitIdx
        else:
            for errorAtomIdx in np.argsort(errorArray)[::-1]:
                errorUnitIdx = self.U.get_unit_index_from_atom_index(errorAtomIdx)
                if newEggLevelArray[errorUnitIdx] == minLevel:
                    if newEggLevelArray[errorUnitIdx] < self.MaxEggLevel:
                        newEggLevelArray[errorUnitIdx] += 1
                        return newEggLevelArray, errorAtomIdx, errorUnitIdx
        return False, False, False

    ###
    def update_egg_with_level_expand_plus_merge(self, errorArray: np.ndarray):
        updateMode = "LXM"
        newMergeIdxLsts, errorAtomIdx, errorUnitIdx = \
                self.get_new_merge_index_list_dict_if_center_merge_and_return_false_when_fail(errorArray)
        newEggLevelArray, errorAtomIdx, errorUnitIdx = \
                self.get_new_egg_level_array_if_unit_wise_expand_and_return_false_when_fail(errorArray)
        if type(newMergeIdxLsts) == type(False) and type(newEggLevelArray) == type(False):
            print_xo_log(" Warning! No level update can be done for any atom.")
            sys.exit(0)
        elif type(newMergeIdxLsts) == type(False) and type(newEggLevelArray) != type(False):
            self.EggLevelArray = newEggLevelArray
            print_xo_log(" {0} Level Expand mode used for unit {1} atom {2} in the next optimization round."\
                    .format(updateMode, errorUnitIdx, errorAtomIdx))
        elif type(newMergeIdxLsts) != type(False) and type(newEggLevelArray) == type(False):
            self.MergeIdxLsts = newMergeIdxLsts
            print_xo_log(" {0} Level Merge mode used for unit {1} atom {2} in the next optimization round."\
                    .format(updateMode, errorUnitIdx, errorAtomIdx))
        else:
            maxHvAtmInEggAfterMerge = self.get_max_egg_size(self.EggLevelArray, newMergeIdxLsts)
            maxHvAtmInEggAfterExpand = self.get_max_egg_size(newEggLevelArray, self.MergeIdxLsts)
            if maxHvAtmInEggAfterMerge < maxHvAtmInEggAfterExpand:
                self.MergeIdxLsts = newMergeIdxLsts
                print_xo_log(" {0} Level Merge mode selected for unit {1} atom {2} in the next optimization round."\
                        .format(updateMode, errorUnitIdx, errorAtomIdx))
            else:
                self.EggLevelArray = newEggLevelArray
                print_xo_log(" {0} Level Expand mode selected for unit {1} atom {2} in the next optimization round."\
                        .format(updateMode, errorUnitIdx, errorAtomIdx))

    ##
    def get_new_merge_index_list_dict_if_center_merge_and_return_false_when_fail(self, errorArray: np.ndarray) -> dict:
        newMergeIdxLsts = deepcopy(self.MergeIdxLsts)
        for centerAtomIdx in np.argsort(errorArray)[::-1]:
            centerUnitIdx = self.U.get_unit_index_from_atom_index(centerAtomIdx)
            iLevel = self.EggLevelArray[centerUnitIdx]
            # print("DU", centerUnitIdx, errorArray[centerAtomIdx], "L", iLevel)
            if iLevel < 0:
                neighborUnitIdxLst = []
            else:
                neighborUnitIdxLst = \
                        list(self.U.UnitNeighborIdxSetLstsAsLevelLst[iLevel][centerUnitIdx][0] - {centerUnitIdx})
            sortedNeighborUnitIdxLst = \
                    [neighborUnitIdxLst[i] for i in np.argsort(errorArray[neighborUnitIdxLst])[::-1]]
            for neighborUnitIdx in sortedNeighborUnitIdxLst:
                if neighborUnitIdx not in newMergeIdxLsts[centerUnitIdx]:
                    newMergeIdxLsts[centerUnitIdx].append(neighborUnitIdx)
                    newMergeIdxLsts[neighborUnitIdx].append(centerUnitIdx)
                    return newMergeIdxLsts, centerAtomIdx, centerUnitIdx
        return False, False, False

    ##
    def get_new_egg_level_array_if_unit_wise_expand_and_return_false_when_fail(self, errorArray: np.ndarray) -> tuple:
        newEggLevelArray = deepcopy(self.EggLevelArray)
        for errorAtomIdx in np.argsort(errorArray)[::-1]:
            errorUnitIdx = self.U.get_unit_index_from_atom_index(errorAtomIdx)
            if newEggLevelArray[errorUnitIdx] < self.MaxEggLevel:
                newEggLevelArray[errorUnitIdx] += 1
                return newEggLevelArray, errorAtomIdx, errorUnitIdx
        return False, False, False
    
    ##
    def get_max_egg_size(self, eggLevelArray: np.ndarray, mergeIdxLsts: dict) -> int:
        # print_xo_log("Degg", eggLevelArray)
        # print_xo_log("Dmc ", self.MergeCenterLevels)
        # print_xo_log("Dmi ", mergeIdxLsts)
        eggUnitIdxSets = self.get_refined_xo2_full_egg_unit_index_set_dict_from_egg_level_array_and_merge_dict(\
                eggLevelArray, mergeIdxLsts)
        eggUnitIdxSetInLst = self.get_unit_index_set_list_with_full_removed(eggUnitIdxSets)
        heavyAtomArray, _ = self.U.get_number_of_atom_arrays_of_unit_index_set_list_from_gjf_file(\
                self.GjfFile, eggUnitIdxSetInLst)
        return int(np.max(heavyAtomArray))
    
    #
    def get_unit_index_set_list_with_full_removed(self, eggUnitIdxSets: dict) -> list:
        eggUnitIdxSetInLst = []
        for eggName in eggUnitIdxSets.keys():
            if eggName != "full":
                eggUnitIdxSetInLst.append(eggUnitIdxSets[eggName])
        return eggUnitIdxSetInLst
    
    ###
    def update_egg_by_unit_expand(self, errorArray: np.ndarray):
        for errorAtomIdx in np.argsort(errorArray)[::-1]:
            errorUnitIdx = self.U.get_unit_index_from_atom_index(errorAtomIdx)
            if self.EggLevelArray[errorUnitIdx] < self.MaxEggLevel:
                self.EggLevelArray[errorUnitIdx] += 1
                return
        print_xo_log("Warning! No level update can be done for any atom.")
        sys.exit(0)

    ###
    def update_egg_by_level_expand(self, errorArray: np.ndarray):
        oldEggLevelArray = deepcopy(self.EggLevelArray)
        minLevel, maxLevel = np.min(oldEggLevelArray), np.max(oldEggLevelArray)
        if maxLevel == minLevel:
            for errorAtomIdx in np.argsort(errorArray)[::-1]:
                errorUnitIdx = self.U.get_unit_index_from_atom_index(errorAtomIdx)
                self.EggLevelArray[errorUnitIdx] += 1
                self.EggLevelArray[self.EggLevelArray > self.U.MaxEggLevel] = self.U.MaxEggLevel
                if not np.all(self.EggLevelArray == oldEggLevelArray):
                    return
        else:
            for errorAtomIdx in np.argsort(errorArray)[::-1]:
                errorUnitIdx = self.U.get_unit_index_from_atom_index(errorAtomIdx)
                if oldEggLevelArray[errorUnitIdx] == minLevel:
                    self.EggLevelArray[errorUnitIdx] += 1
                    self.EggLevelArray[self.EggLevelArray > self.U.MaxEggLevel] = self.U.MaxEggLevel
                    if not np.all(self.EggLevelArray == oldEggLevelArray):
                        return
        print_xo_log("Warning! No level update can be done for any atom.")
        sys.exit(0)
    
    def level_up_for_selected_unit_and_return_true_if_success(self, errorUnitIdx: int,
                                                              oldEggLevelArray: np.ndarray) -> bool:
        self.EggLevelArray[errorUnitIdx] += 1
        self.EggLevelArray[self.EggLevelArray > self.U.MaxEggLevel] = self.U.MaxEggLevel
        if not np.all(self.EggLevelArray == oldEggLevelArray):
            return True
        else:
            return False

    # def upgrade_the_level_for_unit(self, levelUpArray: np.ndarray, errorUnitIdx: int) -> np.ndarray:
    #     neighborUnitIdxArray = np.array(list(self.U.UnitLayerSets[1][errorUnitIdx]))
    #     neighborLevelArray = self.EggLevelArray[neighborUnitIdxArray]
    #     iLevel = self.EggLevelArray[errorUnitIdx]
    #     minLevel = np.min(neighborLevelArray)
    #     if iLevel <= minLevel:
    #         levelUpArray[errorUnitIdx] = True
    #     else:
    #         levelUpArray[neighborUnitIdxArray[neighborLevelArray == minLevel]] = True
    #     return levelUpArray

    #### ^ back
    def update_all_unit_fragmentation(self, errorArray: np.ndarray):
        errorAtomIdxArray = np.argwhere(errorArray > self.XoErrorThreshold).T[0]
        errorUnitIdxArray = [self.U.get_unit_index_from_atom_index(i) for i in errorAtomIdxArray]
        levelUpArray = np.array([False] * self.U.N)
        for iUnit in errorUnitIdxArray:
            neighborUnitIdxArray = list(self.U.UnitLayerSets[1][iUnit])
            neighborLevelArray = self.EggLevelArray[neighborUnitIdxArray]
            iLevel = self.EggLevelArray[iUnit]
            minLevel = np.min(neighborLevelArray)
            if iLevel <= minLevel:
                levelUpArray[iUnit] = True
            else:
                levelUpArray[neighborUnitIdxArray[neighborLevelArray == minLevel]] = True
        self.EggLevelArray[levelUpArray] += 1
        self.EggLevelArray[self.EggLevelArray > self.U.MaxEggLevel] = self.U.MaxEggLevel

    ####
    def print_optimization_round_information(self, iOptRound: int, errorArray: np.ndarray, eggUnitIdxSetInLst: list):
        print_xo_log("------------------------- optimize round", iOptRound, "--------------------------")
        self.print_information_of_error(errorArray)
        self.print_information_of_eggs(eggUnitIdxSetInLst)
        self.print_information_of_mergence()
        self.print_information_of_error_atoms(errorArray)
        print_xo_log("---------------------------------------------------------------------")

    ###
    def print_information_of_error(self, errorArray: np.ndarray):
        errorArray = np.abs(errorArray)
        print_xo_log(" sum error = {:20.10f}".format(np.sum(errorArray)))
        print_xo_log(" max error = {:20.10f}".format(np.max(errorArray)))
        print_xo_log(" rms error = {:20.10f}".format(np.sqrt(np.mean(errorArray**2))))
        print_xo_log(" avg error = {:20.10f}".format(np.mean(errorArray)))
        print_xo_log(" std error = {:20.10f}".format(np.std(errorArray)))

    ###
    def print_information_of_eggs(self, eggUnitIdxSetInLst: list):
        self.print_unit_egg_information()
        heavyAtomArray, _ = self.U.get_number_of_atom_arrays_of_unit_index_set_list_from_gjf_file(\
                self.GjfFile, eggUnitIdxSetInLst)
        print_xo_log(" max number of heavy atom =", np.max(heavyAtomArray))
        print_xo_log(" avg number of heavy atom = {0:.2f}".format(np.mean(heavyAtomArray)))
        print_xo_log(" std number of heavy atom = {0:.2f}".format(np.std(heavyAtomArray)))
        print_xo_log(" heavy atom numbers =", ", ".join([str(i) for i in heavyAtomArray]))

    ##
    def print_unit_egg_information(self):
        if self.UpdateMode == "UXNM":
            self.print_uxnm_egg_information()
        elif self.UpdateMode == "UX":
            self.print_ux_egg_information()
    
    #
    def print_uxnm_egg_information(self):
        if self.NeighborType == "D":
            print_xo_log(" egg, merge center, and merge neighbor levels & radii =")
        else:
            print_xo_log(" egg, merge center, and merge neighbor levels =")
        for unitIdx in sorted(self.MergeCenterLevels.keys()):
            if self.NeighborType == "D":
                eR = self.U.NeighborRadiusDictPerLevel[self.EggLevelArray[unitIdx]][unitIdx]
                if self.MergeCenterLevels[unitIdx] < 0:
                    mcR = 0.0
                else:
                    mcR = self.U.NeighborRadiusDictPerLevel[self.MergeCenterLevels[unitIdx]][unitIdx]
                if self.MergeNeighborLevels[unitIdx] < 0:
                    mnR = 0.0
                else:
                    mnR = self.U.NeighborRadiusDictPerLevel[self.MergeNeighborLevels[unitIdx]][unitIdx]
                print_xo_log(" - Unit {0:>4d} at Level {1:2d} {2:2d} {3:2d} {4:4.1f} {5:4.1f} {6:4.1f}".format(\
                        unitIdx+1, self.EggLevelArray[unitIdx], self.MergeCenterLevels[unitIdx], \
                        self.MergeNeighborLevels[unitIdx], eR, mcR, mnR))
            elif self.NeighborType[0] == "C":
                print_xo_log(" - Unit {0:>4d} at Level {1:2d} {2:2d} {3:2d}".format(\
                        unitIdx+1, self.EggLevelArray[unitIdx], self.MergeCenterLevels[unitIdx], \
                        self.MergeNeighborLevels[unitIdx]))
            else:
                print_error_and_exit("Invalid neighbor type =", self.NeighborType)

    #
    def print_ux_egg_information(self):
        if self.NeighborType == "D":
            print_xo_log(" egg level & radius =")
        else:
            print_xo_log(" egg level =")
        for unitIdx in sorted(self.MergeCenterLevels.keys()):
            if self.NeighborType == "D":
                eR = self.U.NeighborRadiusDictPerLevel[self.EggLevelArray[unitIdx]][unitIdx]
                print_xo_log(" - Unit {0:>4d} at Level {1:2d} {2:4.1f}".format(\
                        unitIdx+1, self.EggLevelArray[unitIdx], eR))
            elif self.NeighborType[0] == "C":
                print_xo_log(" - Unit {0:>4d} at Level {1:2d}".format(unitIdx+1, self.EggLevelArray[unitIdx]))
            else:
                print_error_and_exit("Invalid neighbor type =", self.NeighborType)
    
    ###
    def print_information_of_mergence(self):
        print_xo_log(" mergence:")
        for centerIdx in sorted(self.MergeIdxLsts.keys()):
            mergedSerialLst = sorted([idx + 1 for idx in self.MergeIdxLsts[centerIdx] if idx > centerIdx])
            mergedSerialStrLst = [str(i) for i in mergedSerialLst]
            if len(mergedSerialStrLst) > 0:
                print_xo_log(" - unit {0} merges with {1}".format(centerIdx+1, " ".join(mergedSerialStrLst)))

    ###
    def print_information_of_error_atoms(self, errorArray: np.ndarray):
        nErrorAtom = 20
        errorAtomIdxArray = np.argsort(errorArray)[0:nErrorAtom]
        print_xo_log(" {:<5d} atoms have the largest force error:".format(len(errorAtomIdxArray)))
        if len(errorAtomIdxArray) != 0:
            print_xo_log("   ", " ".join([str(iAtom) for iAtom in errorAtomIdxArray]))
            errorUnitIdxArray = self.U.get_unit_index_list_from_atom_index_list(errorAtomIdxArray)
            print_xo_log(" , and they are distributed in {:5d} units:".format(len(errorUnitIdxArray)))
            print_xo_log("   ", " ".join([str(iUnit) for iUnit in errorUnitIdxArray]))
        print_xo_log("---------------------------------------------------------------------")
    ##### ----- #####


class ConnectivityCalculator(XoSetup, XoCalculator, XoCollector):
    def __init__(self, gjfFile: str, unitFile:str, connectivityFile: str, subFile: str, fragmentDir: str, \
                 queueName: str, nProc: str, nJob: int, \
                 pairDist: float, bufferRangeDist: float, errorType: str, \
                 backChargeRange=0.0, backChargeFile="backChargeFile"):
        XoGeneral.__init__(self)
        # input parameters
        self.GjfFile = gjfFile
        self.QmManager = QMManager(self.ConfigFile, gjfFile, queueName, nProc=nProc)
        self.get_standard_orientation(gjfFile, self.GjfFile, self.QmManager)
        self.UnitFile = unitFile
        self.ConnectivityFile = connectivityFile
        self.SubFile = subFile
        self.FragmentDir = fragmentDir
        self.QueueName = queueName
        self.NProc = nProc
        self.NJob = nJob
        self.PairDist = pairDist
        self.BufferRangeDist = bufferRangeDist
        self.ErrorType = self.get_valid_error_type(errorType)
        self.BackChargeRange = backChargeRange
        self.BackChargeFile = backChargeFile
        # derived parameters
        self.ElementSpins = dict()
        self.TotAtom, self.AtomIdxLstOfUnitLst, self.UnitIdxLst, self.UnitChargeLst, self.UnitSpinLst = \
                read_unit_file(unitFile)
        self.Mem = self.get_mem_from_nproc(nProc)
        self.Charge, self.Spin, self.Xyz, self.ElementLst, self.BondOrders = \
            read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(self.GjfFile)
        self.BasisSets, self.Ecps = self.read_basis_set_and_ecp_from_genecp_file()
        self.U = Unit(self.UnitFile)
        self.Capper = UnitFragmentCapper(self.GjfFile, unitFile, backChargeRange, backChargeFile)
        self.get_connectivity_file()

    ######
    def get_connectivity_file(self):
        if not is_valid_connectivity_file(self.ConnectivityFile):
            print_xo_log("----------------------------------------------------------------------")
            print_xo_log(" As no connectivity file is found, calculating connectivity ...")
            self.calculate_and_write_connectivity_file()
            print_xo_log(" Calculation finished.")
            print_xo_log("----------------------------------------------------------------------")
    
    #####
    def calculate_and_write_connectivity_file(self):
        unitDistNeighbors = self.get_unit_distance_neighbors()
        connectedPairLst, weightLstOfEachPair, methodLevelLstOfEachPair, unitIdxLstOfEachPair, calcPathLstOfEachPair, \
                regionalIdxLstOfEachPair = self.get_connected_pair_property_list(unitDistNeighbors)
        print_xo_log(" - writing pair weight file ...")
        self.write_pair_weight_file(connectedPairLst, weightLstOfEachPair, calcPathLstOfEachPair)
        print_xo_log(" - writing path alias file ...")
        unitPath2WorkPath = self.get_path_alias_dict_and_write_to_file(calcPathLstOfEachPair)
        print_xo_log(" - writing fragment input file of each fragment ...")
        self.write_pairs_fragment_files(methodLevelLstOfEachPair, unitIdxLstOfEachPair, calcPathLstOfEachPair, \
                "force", unitPath2WorkPath)#todo
        print_xo_log(" - Calculating force of each fragment ...")
        XoCalculator.calculate_fragment_from_input(self, self.GjfFile, self.QueueName, self.NJob)
        connectivityLst = self.get_pair_error_list(weightLstOfEachPair, calcPathLstOfEachPair, \
                                                   regionalIdxLstOfEachPair, unitPath2WorkPath)
        print_xo_log(" - writing connectivity file ...")
        self.write_connectivity_file(connectedPairLst, connectivityLst)

    ####
    def get_unit_distance_neighbors(self) -> dict:
        unitDistNeighbors = dict()
        for iUnit in self.U.UnitIdxLst:
            unitDistNeighbors[iUnit] = [iUnit]
        for iUnit in self.U.UnitIdxLst:
            for jUnit in self.U.UnitIdxLst:
                if iUnit < jUnit:
                    if self.if_minimal_inter_atom_distance_below_d(self.U.AtomIdxUnitLst[iUnit], 
                                                                   self.U.AtomIdxUnitLst[jUnit], 
                                                                   self.BufferRangeDist):
                        unitDistNeighbors[iUnit].append(jUnit)
                        unitDistNeighbors[jUnit].append(iUnit)
        return unitDistNeighbors
    
    def if_minimal_inter_atom_distance_below_d(self, atomLst1: list, atomLst2: list, d: float) -> bool:
        for i in atomLst1:
            for j in atomLst2:
                if self.calculate_array_distance(self.Xyz[i], self.Xyz[j]) <= d:
                    return True
        return False
    
    def calculate_array_distance(self, array1: np.ndarray, array2: np.ndarray) -> float:
        return np.sqrt(np.sum((array1 - array2)**2))
    
    ####
    def get_connected_pair_property_list(self, unitDistNeighbors: dict):
        connectedPairLst, weightLstOfEachPair, methodLevelLstOfEachPair = [], [], []
        unitIdxLstOfEachPair, calcPathLstOfEachPair, regionalIdxLstOfEachPair = [], [], []
        for iUnit, aUnitLst in enumerate(self.U.AtomIdxUnitLst):
            for jUnit, bUnitLst in enumerate(self.U.AtomIdxUnitLst):
                if iUnit < jUnit:
                    if self.if_minimal_inter_atom_distance_below_d(aUnitLst, bUnitLst, self.PairDist):
                        # print("Du", iUnit, jUnit, self.PairDist)
                        connectedPairLst.append([iUnit, jUnit])
                        weightLst, methodLevelLst, unitIdxLst, calcPathLst, regionalIdxLst = \
                                self.get_pair_properties(iUnit, jUnit, unitDistNeighbors)
                        weightLstOfEachPair.append(weightLst)
                        methodLevelLstOfEachPair.append(methodLevelLst)
                        unitIdxLstOfEachPair.append(unitIdxLst)
                        calcPathLstOfEachPair.append(calcPathLst)
                        regionalIdxLstOfEachPair.append(regionalIdxLst)
        return connectedPairLst, weightLstOfEachPair, methodLevelLstOfEachPair, \
               unitIdxLstOfEachPair, calcPathLstOfEachPair, regionalIdxLstOfEachPair

    ### only for XO2
    def get_pair_properties(self, iUnit: int, jUnit: int, unitDistNeighbors: dict) -> tuple:
        unitAorBLst = sorted(set(unitDistNeighbors[iUnit]) | set(unitDistNeighbors[jUnit]))
        unitALst = sorted(unitDistNeighbors[iUnit])
        unitBLst = sorted(unitDistNeighbors[jUnit])
        unitAandBLst = sorted(set(unitDistNeighbors[iUnit]) & set(unitDistNeighbors[jUnit]))
        if len(unitAandBLst) > 0:
            weightLst = [1, -1, -1, 1, -1, 1, 1, -1]
            methodLevelLst = ["H"] * 4 + ["L"] * 4
            unitIdxLstInLst = [unitAorBLst, unitALst, unitBLst, unitAandBLst]
        else:
            weightLst = [1, -1, -1, -1, 1, 1]
            methodLevelLst = ["H"] * 3 + ["L"] * 3
            unitIdxLstInLst = [unitAorBLst, unitALst, unitBLst]
        calcPathLst = self.get_calculate_path_list_from_unit_list(unitIdxLstInLst)
        regionalIdxLst = self.get_regional_index_list_from_unit_list(unitIdxLstInLst)
        unitIdxLstInLst = unitIdxLstInLst * len(self.MethodLevelLst)
        return weightLst, methodLevelLst, unitIdxLstInLst, calcPathLst, regionalIdxLst
    
    ###
    def get_calculate_path_list_from_unit_list(self, unitLstInLst: list) -> list:
        unitDirLst = []
        for unitLst in unitLstInLst:
            unitDirLst.append(get_dir_name_from_index_list("U", unitLst))
        calcPathLst = []
        for uLevel in self.MethodLevelLst:
            for uDir in unitDirLst:
                calcPathLst.append(os.path.join(self.FragmentDir, uDir, uLevel))
        return calcPathLst

    def get_regional_index_list_from_unit_list(self, unitLstInLst: list) -> list:
        #NOTE! unitLstInLst[0] must be AorB
        globalAtomIdxLstOfAorB = list(flatten([self.U.AtomIdxUnitLst[iUnit] for iUnit in unitLstInLst[0]]))
        globalAtomIdx2regionalAtomIdx = dict()
        for iRegional, iGlobal in enumerate(globalAtomIdxLstOfAorB):
            globalAtomIdx2regionalAtomIdx[iGlobal] = iRegional
        #
        regionalIdxLst = []
        for unitLst in unitLstInLst:
            tmpGlobalAtomIdxLst = list(flatten([self.U.AtomIdxUnitLst[iUnit] for iUnit in unitLst]))
            regionalIdxLst.append([globalAtomIdx2regionalAtomIdx[iGlobal] for iGlobal in tmpGlobalAtomIdxLst])
        return regionalIdxLst

    ####
    def write_pair_weight_file(self, connectedPairLst: list, weightLstOfEachPair: list, calcPathLstOfEachPair: list):
        with open(self.PairWeightFile, "w") as wfl:
            for uPair, uWeightLst, uCalcPathLst in zip(connectedPairLst, weightLstOfEachPair, calcPathLstOfEachPair):
                print("pair", uPair[0], uPair[1], file=wfl)
                for uWeight, uDir in zip(uWeightLst, uCalcPathLst):
                    print("{:2d} {:s}".format(uWeight, uDir), file=wfl)

    ####
    def get_path_alias_dict_and_write_to_file(self, calcPathLstOfEachPair: list) -> dict:
        uniqueCalcPathLst = []
        for calcPathLst in calcPathLstOfEachPair:
            for uCalcPath in calcPathLst:
                if uCalcPath not in uniqueCalcPathLst:
                    uniqueCalcPathLst.append(uCalcPath)
        return append_path_alias_file_and_return_new_path_alias_dict(self.PathAliasFile, uniqueCalcPathLst)
    
    ####
    def write_fragment_file(self, methodLevelLstOfEachPair: list, unitIdxLstOfEachPair: list, \
                            calcPathLstOfEachPair: list, option: str, unitPath2WorkPath: dict) -> list:
        uniqueCalcPathLst = []
        for unitIdxLstInLst, calcPathLst, levelLst in zip(unitIdxLstOfEachPair, calcPathLstOfEachPair, \
                                                          methodLevelLstOfEachPair):
            for unitIdxLst, uCalcPath, uLevel in zip(unitIdxLstInLst, calcPathLst, levelLst):
                if uCalcPath not in uniqueCalcPathLst:
                    inputFile = "fragment" + self.QmManager.InputExtentions[uLevel]
                    uniqueCalcPathLst.append(uCalcPath)
                    mk_serial_dir(unitPath2WorkPath[uCalcPath])
                    fragCharge, fragSpin, fragXyzArray, fragElementLst, fragBondOrders, fragXyzChargeArray, _ = \
                            self.Capper.cap_fragment_from_unit_index_list(unitIdxLst)
                    self.QmManager.write_fragment_input_file(\
                            inputFile, uLevel, self.NProc, self.Mem, \
                            self.Methods[uLevel], option, fragCharge, fragSpin, \
                            fragElementLst, fragXyzArray, fragBondOrders, fragXyzChargeArray, \
                            self.BasisSets[uLevel], self.Ecps[uLevel], \
                            self.GaussianExtraInputs[uLevel], self.ElementSpins, [])
    
    #####
    def get_pair_error_list(self, weightLstOfEachPair: list, 
                            calcPathLstOfEachPair: list, regionalIdxLstOfEachPair: list, 
                            unitPath2WorkPath: dict) -> list:
        if self.ErrorType == "F":
            errorLst = self.get_pair_max_force_error_list(weightLstOfEachPair, calcPathLstOfEachPair, \
                                                          regionalIdxLstOfEachPair, unitPath2WorkPath)
        else:
            print_error_and_exit("Unsupported connectivity type =", self.ErrorType)
        return errorLst

    ####
    def get_pair_max_force_error_list(self, weightLstOfEachPair: list, calcPathLstOfEachPair: list, \
                                      regionalIdxLstOfEachPair: list, unitPath2WorkPath: dict) -> list:
        MAXATOM = 50001
        maxForceErrorLst = []
        for weightLst, unitPathLst, regionalIdxLst in zip(weightLstOfEachPair, calcPathLstOfEachPair, 
                                                          regionalIdxLstOfEachPair):
            maxAtom = -1
            sumForce = np.zeros((MAXATOM, 3))
            for weight, unitPath, regionalAtomIdxLst in zip(weightLst, unitPathLst, regionalIdxLst):
                localAtomIdxLst = list(range(len(regionalAtomIdxLst)))
                level = self.QmManager.get_method_level_from_unit_or_work_path(unitPath)
                outFile = self.FragmentFilePref + self.QmManager.OutputExtentions[level]
                force = self.QmManager.read_force_function_dict()[level](\
                        os.path.join(unitPath2WorkPath[unitPath], outFile))
                if np.max(regionalAtomIdxLst) > maxAtom:
                    maxAtom = np.max(regionalAtomIdxLst)
                    if maxAtom > MAXATOM - 1:
                        print_error_and_exit("Too many atoms in the system. Please increase the value of MAXATOM \
                                in the get_pair_force_error_list subroutine.")
                sumForce[regionalAtomIdxLst, :] += force[localAtomIdxLst, :] * weight
            maxForceError = np.max(np.sqrt(np.sum(sumForce*sumForce, axis=1)))
            maxForceErrorLst.append(maxForceError)
        return maxForceErrorLst
    
    #####
    def write_connectivity_file(self, connectedPariLst: list, connectivityLst: list):
        with open(self.ConnectivityFile, "w") as wfl:
            for uPair, uConnectivity in zip(connectedPariLst, connectivityLst):
                print("{:5d} {:5d} {:20.12f}".format(uPair[0], uPair[1], uConnectivity), file=wfl)

class GeneralXoProcessor(XoSetup, XoCalculator, XoCollector):
    def __init__(self, gjfFile: str, subFile: str, fragmentDir: str, \
                 queueName: str, nProc: str, mem: str, nJob: int, isSymm=False, \
                 backChargeRange=0.0, backChargeFile="backChargeFile", useStandardOrientation=True):
        XoGeneral.__init__(self)
        # input properties
        self.QmManager = QMManager(self.ConfigFile, gjfFile, queueName, nProc=nProc)
        self.get_standard_orientation(gjfFile, self.GjfFile, self.QmManager, useStandardOrientation)
        self.SubFile = subFile
        self.FragmentDir = fragmentDir
        self.QueueName = queueName
        self.NProc = str(nProc)
        self.Mem = mem
        self.NJob = nJob
        self.IsSymm = isSymm
        self.BackChargeRange = backChargeRange
        self.BackChargeFile = backChargeFile
        # derived properties
        self.ElementSpins = dict()
        self.Methods, self.ExecutableCommands, self.Executables, self.GaussianExtraInputs \
                = read_data_from_config_file(self.ConfigFile)
        self.MethodLevelLst = sorted(self.Methods.keys())
        self.BasisSets, self.Ecps = self.read_basis_set_and_ecp_from_genecp_file()
        # self.MemLine, self.NprocLine, self.ChkLine, self.RouteLine = \
        #         read_resource_and_route_from_gjf_file(self.GjfFile)
        self.Nproc = self.refine_nproc_with_nProc(nProc)
        self.Mem = self.refine_mem_with_nProc(mem)
        # self.Option = self.get_xo_option_from_route_line(self.RouteLine)
        self.Option = ""
        _, _, _, self.ElementLst, _ = read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(self.GjfFile)
        self.TotAtom = len(self.ElementLst)
        self.ResultType = get_result_type_from_input_file(self.GjfFile)
    
    ##### ----- #####
    #####
    def derive_fragments_and_write_weight_path_alias_and_fragment_files(self, isUnit=True):
        print_xo_log(" Preparing for fragment files ...")
        self.FragmentDir1stChar = "U" if isUnit else "A"
        fragmentPathLst = self.write_weight_file_and_return_fragment_path_list()
        _ = append_path_alias_file_and_return_new_path_alias_dict(self.PathAliasFile, fragmentPathLst)
        self.write_fragment_files_from_weight_and_path_alias_files_with_lines(isUnit)
    
    ####
    def write_fragment_files_from_weight_and_path_alias_files_with_lines(self, isUnit: bool):
        # fragmentFiles = inputFile + globalIdxFile
        rootPath = os.getcwd()
        _, uniqueUnitPathLst = read_weight_file(self.WeightFile)
        _, unitPath2WorkPath = read_path_alias_file(self.PathAliasFile)
        for uUnitPath in uniqueUnitPathLst:
            #uUnitPath = .unit/U1-3_5/H
            workPath = unitPath2WorkPath[uUnitPath]
            mk_serial_dir(workPath)
            os.chdir(workPath)
            level, idxLst = get_method_level_and_index_list_from_calc_path(uUnitPath)
            inputFile = "fragment" + self.QmManager.InputExtentions[level]
            outputFile = "fragment" + self.QmManager.OutputExtentions[level]
            if type(self.Methods[level]) == type("a"):
                routeLine = "#p " + self.Methods[level] + " " + self.Option
            else:
                routeLine = "#p " + self.Option
            if isUnit:
                fragCharge, fragSpin, fragXyzArray, fragElementLst, fragBondOrders, fragXyzChargeArray, \
                        globalIdxLst = self.Capper.cap_fragment_from_unit_index_list(idxLst)
            else:
                fragCharge, fragSpin, fragXyzArray, fragElementLst, fragBondOrders, fragXyzChargeArray, \
                        globalIdxLst = self.Capper.cap_fragment_from_atom_index_list(idxLst)
            if len(fragXyzChargeArray) > 0:
                routeLine += " charge"
            if not os.path.isfile(self.GlobalIdxFile):
                write_global_index_file(self.GlobalIdxFile, globalIdxLst)
            isRun = False
            #todo
            if "CalculateFragmentEveryTime" not in self.__dict__:
                self.CalculateFragmentEveryTime = False
            if self.CalculateFragmentEveryTime:
                isRun = True
            else:
                if not self.QmManager.is_calculation_exit_normally_for_a_type(workPath, outputFile, self.ResultType):
                    isRun = True
            if isRun:
                self.warn_and_remove_fragment_out_file(rootPath, workPath, fragElementLst, 
                                                       isWarn=not self.CalculateFragmentEveryTime)
                self.QmManager.write_fragment_input_file(inputFile, level, self.NProc, self.Mem, \
                                                         self.Methods, self.ResultType, fragCharge, fragSpin, \
                                                         fragElementLst, fragXyzArray, fragBondOrders, \
                                                         fragXyzChargeArray, self.BasisSets, self.Ecps, \
                                                         self.GaussianExtraInputs, self.ElementSpins, globalIdxLst)
            os.chdir(rootPath)
    ##### ----- #####
    
    ##### ----- #####
    #####
    def calculate(self):
        print_xo_log(" Calculating each fragment ...")
        uniqueWorkPathLst = self.get_unfinished_unique_work_path_list()
        if self.QueueName.lower() == "local":
            self.calculate_locally(uniqueWorkPathLst)
        else:
            self.calculate_by_queue(uniqueWorkPathLst)

    ### self.ResultType
    def calculate_locally(self, uniqueWorkPathLst: list):
        XoCalculator.calculate_fragment_locally_from_input(self, uniqueWorkPathLst)

    ###
    def calculate_by_queue(self, uniqueWorkPathLst: list):
        XoCalculator.calculate_fragment_by_queue_from_input(self, self.GjfFile, self.QueueName, \
                                                            uniqueWorkPathLst, self.NJob)
    ##### ----- #####

    ##### ----- #####
    #0
    def collect_result_and_print(self, suffix=""):
        print_xo_log(" Collecting results ...")
        collector = XoCollector(gjfFile=self.GjfFile)
        _ = collector.collect_xo_result()
        print_xo_log(" Writing results ...")
        collector.write_xo_result_to_file(suffix)
    ##### ----- #####
    
    ##### ----- #####
    #0
    def collect_result_list_and_write_gaussian_external_file(self, outputExternalFile: str):
        print_xo_log(" Collecting result list ...")
        collector = XoCollector(gjfFile=self.GjfFile)
        self.XoResultLst = collector.collect_xo_result_list()
        print_xo_log(" Writing result list to external output file ...")
        self.write_result_list_to_gaussian_external_file(outputExternalFile)
    
    #1
    def write_result_list_to_gaussian_external_file(self, outputExternalFile: str):
        # xoResultLst = [energy, force, forceConstant]
        if len(self.XoResultLst) == 1:
            write_gaussian_external_output_file(outputExternalFile, self.XoResultLst[0])
        elif len(self.XoResultLst) == 2:
            write_gaussian_external_output_file(outputExternalFile, self.XoResultLst[0], 
                                                gradArray=-self.XoResultLst[1])
        elif len(self.XoResultLst) == 3:
            write_gaussian_external_output_file(outputExternalFile, self.XoResultLst[0], 
                                                gradArray=-self.XoResultLst[1],
                                                forceConstant2dArray=self.XoResultLst[2])
        else:
            print_error_and_exit("Invalid dimension of XO result list.")
    ##### ----- #####


class UnitBasedXoProcessor(GeneralXoProcessor):
    def __init__(self, gjfFile: str, unitFile: str, eggFile: str, subFile: str, fragmentDir: str, \
                 queueName: str, nProc: str, mem: str, nJob: int, isSymm=False, \
                 backChargeRange=0.0, backChargeFile="", oldWeightFile="", \
                 calculateFragmentEveryTime=False, useStandardOrientation=True):
        GeneralXoProcessor.__init__(self, gjfFile, subFile, fragmentDir, \
                                    queueName, nProc, mem, nJob, isSymm, \
                                    backChargeRange, backChargeFile, useStandardOrientation)
        # input properties
        self.UnitFile = unitFile
        self.EggFile = eggFile
        self.OldWeightFile = oldWeightFile
        self.CalculateFragmentEveryTime = calculateFragmentEveryTime
        # derived properties
        if self.CalculateFragmentEveryTime:
            if os.path.isdir(fragmentDir):
                rmtree(fragmentDir)
            if os.path.isfile(self.WeightFile):
                os.unlink(self.WeightFile)
            if os.path.isfile(self.PairWeightFile):
                os.unlink(self.PairWeightFile)
        self.Capper = UnitFragmentCapper(self.GjfFile, unitFile, backChargeRange, backChargeFile)
        self.EggIdxSets, self.EggMethodLevels, self.EggCharges, self.EggSpins = \
                read_unit_egg_files_and_output_egg_dicts(unitFile, eggFile)
        self.AllIdxSetAtLevels, self.HighEggIdxSets, self.MediumEggIdxSets, self.LowEggIdxSets, \
                self.UniqueLevelLst = self.get_index_set_at_each_level()


class AtomBasedXoProcessor(GeneralXoProcessor):
    def __init__(self, gjfFile: str, schemeFile: str, subFile: str, fragmentDir: str, \
                 queueName: str, nProc: str, mem: str, nJob: int, isSymm=False, \
                 backChargeRange=0.0, backChargeFile=""):
        GeneralXoProcessor.__init__(self, gjfFile, subFile, fragmentDir, queueName, nProc, mem, nJob, isSymm, \
                                    backChargeRange, backChargeFile)
        # input properties
        self.SchemeFile = schemeFile
        # derived properties
        self.Capper = FragmentCapper(schemeFile, self.GjfFile, backChargeRange, backChargeFile)
        self.EggIdxSets, self.EggMethodLevels, self.EggCharges, self.EggSpins = \
                read_simple_XO2_fragment_set_and_level_dict_from_scheme_file(schemeFile)
                #todo ensure all atoms appear in schemefile
        self.AllIdxSetAtLevels, self.HighEggIdxSets, self.MediumEggIdxSets, self.LowEggIdxSets, \
                self.UniqueLevelLst = self.get_index_set_at_each_level()


class UnitEggRefiner(UnitFragmentOptimizer):
    def __init__(self, gjfFile: str, unitFile: str, subFile: str, fragmentDir: str, queueName: str, \
                 nProc: int, nJob: int, mem: str, updateMode: str, maxRound: int, threshold: float, \
                 initRadius: float, maxRadius: float, statRadius: float, \
                 eggFilePref="eggFile", chargeRange=0.0, chargeFile="", oldWeightFile=""):
        XoGeneral.__init__(self)
        # initial properties
        self.GjfFile = gjfFile
        self.UnitFile = unitFile
        self.SubFile = subFile
        self.FragmentDir = fragmentDir
        self.QueueName = queueName
        self.NProc = nProc
        self.NJob = nJob
        self.Mem = mem
        self.UpdateMode = updateMode
        self.MaxRound = maxRound
        self.ForceErrorThreshold = threshold
        self.InitRadius = initRadius
        self.MaxRadius = maxRadius
        self.StatRadius = 0.1 if statRadius==0 else statRadius
        self.EggFilePref = eggFilePref
        self.ChargeRange = chargeRange
        self.ChargeFile = chargeFile
        self.OldWeightFile = oldWeightFile
        # XO setup properties
        self.XoLevel = self.get_xo_level_from_config_file()
        self.Capper = UnitFragmentCapper(gjfFile, unitFile, chargeRange, chargeFile)
        self.Charge, self.Spin, self.Xyz, self.ElementLst, self.BondOrders = \
                read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(gjfFile)
        self.BasisSets, self.Ecps = self.read_basis_set_and_ecp_from_genecp_file()
        self.ResultType = get_result_type_from_input_file(gjfFile)
        self.TotAtom, self.AtomIdxLstOfUnitLst, self.UnitIdxLst, self.UnitChargeLst, self.UnitSpinLst = \
                read_unit_file(unitFile)
        self.QmManager = QMManager(self.ConfigFile, gjfFile, queueName=queueName, nProc=str(nProc), \
                                   resultType="force")
        # derived properties
        self.AtomNeighborIdxSets = self.get_atom_neighbor_set_dict_from_gjf_file(self.GjfFile, self.StatRadius)
        self.RUnitInit = RadiusUnit(gjfFile, unitFile, self.InitRadius)
        self.RUnitMax = RadiusUnit(gjfFile, unitFile, self.MaxRadius)
        self.U = self.RUnitInit
        self.UnitIdxLst = self.RUnitInit.UnitIdxLst
        self.NUnit = self.RUnitInit.N
        self.UnitEggIdxSetsPerLevel = self.get_unit_neighbor_index_set_dict_per_level() #TODO check
        self.FullUnitIdxSet = get_full_index_set_from_dict(self.RUnitInit.InitialRawEggUnitIdxSets)
        self.EggLevelArray = [0 for _ in range(self.NUnit)]
        self.NEggLevel = 2
        self.MaxEggLevel = self.NEggLevel - 1
        self.Collector = None
    
    #-0
    def get_xo_level_from_config_file(self) -> int:
        self.Methods, _, _, _ = read_data_from_config_file(self.ConfigFile)
        uniqueLevelSet = set(self.Methods.keys())
        self.ensure_valid_unique_level_set(uniqueLevelSet)
        return len(uniqueLevelSet)
    
    #-0
    def get_atom_neighbor_set_dict_from_gjf_file(self, gjfFile: str, radius: float) -> dict:
        atomNeighborIdxSets = dict()
        radius2 = radius * radius
        _, _, xyzArray, _, _ = read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(gjfFile)
        for i, uxyz in enumerate(xyzArray):
            atomNeighborIdxSets[i] = np.where(np.sum((xyzArray - uxyz)**2, axis=1) <= radius2)
        return atomNeighborIdxSets
    
    #-0
    def get_unit_neighbor_index_set_dict_per_level(self) -> dict:
        unitEggIdxSetsPerLevel = dict()
        unitEggIdxSetsPerLevel[0] = self.RUnitInit.InitialRawEggUnitIdxSets
        unitEggIdxSetsPerLevel[1] = self.RUnitMax.StuffedRawEggUnitIdxSets
        return unitEggIdxSetsPerLevel
    
    #0
    def refine_high_level_eggs(self):
        print_section_to_xo_log(" Initializing refinement process ...".format(0))
        iRound = 0
        self.EggIdxSets, self.EggMethodLevels = self.get_all_egg_dicts_from_egg_level_array()
        self.write_egg_file_and_set_self_egg_file(self.get_refine_egg_file_name(iRound))
        atomErrorArray, maxError = self.calculate_force_error_and_return_error_array_and_max()
        print_section_to_xo_log(" Initialization finished.")
        self.print_refine_round_information(iRound, atomErrorArray)
        print_section_to_xo_log(" Starting refinement round {0} ...".format(iRound + 1))
        while iRound <= self.MaxRound and maxError >= self.ForceErrorThreshold:
            atomErrorArray, maxError, isChanged = self.update_egg_to_reduce_force_error(atomErrorArray)
            if isChanged:
                iRound += 1
                self.write_egg_file_and_set_self_egg_file(self.get_refine_egg_file_name(iRound))
                self.print_refine_round_information(iRound, atomErrorArray)
            else:
                print_section_to_xo_log(" Normal exit. No force error improvement can be done. Refinement ends.")
                sys.exit(1)
        self.print_normal_exit_information(iRound, maxError)
    
    #1
    def get_all_egg_dicts_from_egg_level_array(self) -> tuple:
        eggIdxSets = self.get_high_level_egg_dict_from_egg_level_array()
        eggIdxSets, eggLevels = self.get_level_dict_and_add_full_unit_index_set_to_low_level_if_need(eggIdxSets)
        return eggIdxSets, eggLevels

    #2
    def get_high_level_egg_dict_from_egg_level_array(self) -> dict:
        eggIdxSets = dict()
        for eggName in self.RUnitInit.InitialRawEggUnitIdxSets.keys():
            centerUnitIdx = get_center_unit_index_from_egg_name(eggName)
            eggLevel = self.EggLevelArray[centerUnitIdx]
            eggIdxSets[eggName] = self.UnitEggIdxSetsPerLevel[eggLevel][eggName]
        return remove_subset_in_dict(eggIdxSets)
    
    #2
    def get_level_dict_and_add_full_unit_index_set_to_low_level_if_need(self, eggIdxSets: dict) -> tuple:
        newEggIdxSets = eggIdxSets
        eggLevels = dict()
        for uName in newEggIdxSets.keys():
            eggLevels[uName] = "H"
        if self.XoLevel == 2:
            newEggIdxSets["full"] = self.FullUnitIdxSet
            eggLevels["full"] = "L"
        return newEggIdxSets, eggLevels
    
    #1
    def write_egg_file_and_set_self_egg_file(self, eggFile: str):
        write_egg_file_from_egg_dicts(eggFile, self.EggIdxSets, self.EggMethodLevels)
        self.EggFile = eggFile
    
    #1
    def get_refine_egg_file_name(self, iRound: int) -> str:
        return "{0}.ir{1:.1f}mr{2:.1f}sr{3:.1f}.round.{4:d}".format(self.EggFilePref, \
               self.InitRadius, self.MaxRadius, self.StatRadius, iRound)
    
    #1
    def calculate_force_error_and_return_error_array_and_max(self):
        print_xo_log(" Generating fragments and write to weight file...")
        self.get_fragment_weight_and_dir_lists_and_write_to_weight_and_path_alias_files()
        print_xo_log(" Writing input calculation files ...")
        self.write_fragment_force_files_from_weight_and_path_alias_files()
        self.write_high_level_full_force_input_file()
        print_xo_log(" Calculating each fragment ...")
        xoCalculator = XoCalculator(self.GjfFile, self.SubFile, self.QueueName, self.NProc, self.NJob, self.GjfFile)
        xoCalculator.calculate_fragment_and_full()
        print_xo_log(" Collecting results ...")
        if self.Collector == None:
            self.Collector = XoCollector(resultType="force", gjfFile=self.GjfFile)
        _ = self.Collector.collect_xo_result()
        _ = self.Collector.read_full_result()
        forceErrorArray = self.Collector.calculate_difference()
        return forceErrorArray, np.max(forceErrorArray)
    
    #1
    def print_refine_round_information(self, iRound: int, errorArray: np.ndarray):
        eggNameLst, eggIdxSetInLst = get_egg_name_and_index_set_lists_at_selected_level(\
                self.EggIdxSets, self.EggMethodLevels, "H")
        # self.print_optimization_round_information(iRound, errorArray, eggIdxSetInLst)
        print_xo_log("---------------------- refinement round", iRound, "----------------------")
        self.print_information_of_error(errorArray)
        print_xo_log(" error:", ", ".join(["{:.6f}".format(error) for error in errorArray]))
        self.print_information_of_eggs(eggIdxSetInLst)
        self.print_information_of_error_atoms(errorArray)
        print_xo_log("---------------------------------------------------------------------")
    
    #1
    def update_egg_to_reduce_force_error(self, oldAtomErrorArray: np.ndarray) -> tuple:
        oldRegionErrorArray, oldMaxRegionError = self.get_region_error_array_and_its_max(oldAtomErrorArray)
        oldErrorSum = np.sum(oldAtomErrorArray)
        iTrial = 1
        for uIdx in np.argsort(oldRegionErrorArray)[::-1]:
            if self.EggLevelArray[uIdx] < self.MaxEggLevel:
                print_xo_log(" --- trial {0} ---".format(iTrial))
                iTrial += 1
                self.EggLevelArray[uIdx] += 1
                self.EggIdxSets, self.EggMethodLevels = self.get_all_egg_dicts_from_egg_level_array()
                self.write_egg_file_and_set_self_egg_file(self.EggFilePref + "_refine.temp")
                newAtomErrorArray, maxError = self.calculate_force_error_and_return_error_array_and_max()
                print_xo_log(" --- error sum:", "{:15.6f}".format(np.sum(newAtomErrorArray)))
                # print_xo_log(" error:", "{:15.6f}".format(np.sum(newAtomErrorArray)), "|", \
                #         ", ".join(["{:.6f}".format(error) for error in newAtomErrorArray]))
                # _, newMaxRegionError = self.get_region_error_array_and_its_max(newAtomErrorArray)
                newErrorSum = np.sum(newAtomErrorArray)
                if newErrorSum <= oldErrorSum:
                    print_xo_log(" --- Unit {0} level up! ---".format(uIdx))
                    return newAtomErrorArray, maxError, True
                else:
                    self.EggLevelArray[uIdx] -= 1
        return oldAtomErrorArray, np.max(oldAtomErrorArray), False
    
    #2
    def get_region_error_array_and_its_max(self, errorArray: np.ndarray) -> float:
        regionErrorLst = []
        for uIdx in self.UnitIdxLst:
            regionError = np.sum(errorArray[self.AtomNeighborIdxSets[uIdx]])
            regionErrorLst.append(regionError)
        regionErrorArray = np.array(regionErrorLst)
        return regionErrorArray, np.max(regionErrorArray)
    
    #1
    def print_normal_exit_information(self, iRound: int, maxError: float):
        if iRound > self.MaxRound:
            print_xo_log("Normal exit. Max round {0} is reached. Refinement ends.".format(self.MaxRound))
            sys.exit(1)
        if maxError < self.ForceErrorThreshold:
            print_xo_log("Normal exit. Force error threshold {0} is reached: {1}. Refinement ends.".format(\
                    self.ForceErrorThreshold, maxError))
            sys.exit(1)


class Embedded2bodyForceErrorCalculator(UnitEggRefiner):
    def __init__(self, gjfFile: str, unitFile: str, radius: float,\
                 fragmentDir: str, queueName: str, subFile: str, nProc: str, mem: str, 
                 nJob: int, eggLevelArray=np.array([])):
        XoGeneral.__init__(self)
        self.GjfFile = gjfFile
        self.UnitFile = unitFile
        self.Radius = radius
        self.FragmentDir = fragmentDir
        self.QueueName = queueName
        self.SubFile = subFile
        self.NProc = str(nProc)
        self.Mem = mem
        self.NJob = nJob
        self.EggLevelArray = eggLevelArray
        #
        self.OldWeightFile = ""
        self.AdditionalUnitFile = unitFile + ".add"
        self.MINVAL = 1e-30
        #
        self.RadUnit = RadiusUnit(gjfFile, unitFile, radius)
        self.RadUnit.write_merged_unit_pairs_to_file(self.AdditionalUnitFile)
        self.EggFileIsolated = self.EggFile + ".isolate"
        self.Setup = XoSetup(gjfFile, unitFile, "", fragmentDir, nProc, eggLevelArray, mem=mem, neighborType="D")
        self.Calculator = XoCalculator(gjfFile, subFile, queueName, nProc, nJob, gjfFile)
        self.Collector = XoCollector(resultType="force", gjfFile=gjfFile)
    
    #0
    def calculate_and_collect_linking_bonus_per_pair(self):
        self.calculate_isolated_xo_force_and_full_force()
        self.calculate_2_body_connected_xo_force()
        self.get_linking_bonus_2d_array_from_forces()
    
    #1
    def calculate_isolated_xo_force_and_full_force(self):
        extension = ".isolated"
        newWeightFile = self.WeightFile + extension
        self.Setup.set_weight_file(newWeightFile)
        self.Calculator.set_weight_file(newWeightFile)
        self.Collector.set_weight_file(newWeightFile)
        print_xo_log(" -------- isolated units & full system --------")
        print_xo_log(" Generating eggs and write to egg file ...")
        rawEggIdxSets = self.RadUnit.InitialRawEggUnitIdxSets
        eggIdxSets, eggMethodLevels = get_refined_xo2_full_egg_index_set_and_method_level_dicts(rawEggIdxSets)
        write_egg_file_from_egg_dicts(self.EggFileIsolated, eggIdxSets, eggMethodLevels)
        print_xo_log(" Generating fragments and write to weight file ...")
        if os.path.isfile(self.OldWeightFile):
            os.unlink(self.OldWeightFile)
        self.Setup.set_egg_unit_idx_set_dict(eggIdxSets)
        self.Setup.set_egg_method_level_dict(eggMethodLevels)
        self.Setup.set_external_egg_file(self.EggFileIsolated)
        self.Setup.get_fragment_weight_and_dir_lists_and_write_to_weight_and_path_alias_files()
        print_xo_log(" Writing input calculation files ...")
        self.Setup.write_fragment_force_files_from_weight_and_path_alias_files()
        self.Setup.write_high_level_full_force_input_file()
        print_xo_log(" Submitting jobs to queue for calculation ...")
        self.Calculator.calculate_fragment_and_full()
        print_xo_log(" Collecting results ...")
        self.XoForceIsolated = self.Collector.collect_xo_result()
        self.FullForce = self.Collector.collect_full_result()
        print_xo_log(" ----------------------------------------------")

    #1
    def calculate_2_body_connected_xo_force(self):
        self.XoForce2b = self.initialize_unit_2d_dict()
        for iUnit, jUnit in self.RadUnit.ConnectedPairLst:
            newUnitIdx = renew_additional_unit_file_and_return_index_of_new_unit(\
                         self.AdditionalUnitFile, self.UnitFile, [iUnit, jUnit])
            extension = self.get_file_extension([iUnit, jUnit])
            newWeightFile = self.WeightFile + extension
            newEggFile = self.EggFile + extension
            self.Setup.set_weight_file(newWeightFile)
            self.Calculator.set_weight_file(newWeightFile)
            self.Collector.set_weight_file(newWeightFile)
            print_xo_log(" -------- system with unit {0} + {1} --------".format(iUnit+1, jUnit+1))
            print_xo_log(" Generating eggs and write to egg file ...")
            rawEggIdxSets = self.RadUnit.get_egg_index_set_dict_after_merge([iUnit, jUnit], newUnitIdx)
            #print("D1", iUnit, jUnit)
            #print("D2", rawEggIdxSets)
            #print("D3", newUnitIdx)
            eggIdxSets, eggMethodLevels = get_refined_xo2_full_egg_index_set_and_method_level_dicts(rawEggIdxSets)
            write_egg_file_from_egg_dicts(newEggFile, eggIdxSets, eggMethodLevels)
            print_xo_log(" Generating fragments and write to weight file ...")
            if os.path.isfile(self.OldWeightFile):
                os.unlink(self.OldWeightFile)
            self.Setup.set_egg_unit_idx_set_dict(eggIdxSets)
            self.Setup.set_egg_method_level_dict(eggMethodLevels)
            self.Setup.set_external_egg_file(newEggFile)
            self.Setup.get_fragment_weight_and_dir_lists_and_write_to_weight_and_path_alias_files()
            print_xo_log(" Writing input calculation files ...")
            self.Setup.write_fragment_force_files_from_weight_and_path_alias_files()
            print_xo_log(" Submitting jobs to queue for calculation ...")
            self.Calculator.calculate_fragment_and_full(isFullCalculated=False)
            print_xo_log(" Collecting results ...")
            self.XoForce2b[iUnit][jUnit] = self.Collector.collect_xo_result()
            self.XoForce2b[jUnit][iUnit] = self.XoForce2b[iUnit][jUnit]
            print_xo_log(" ----------------------------------------------")
    
    #2
    def initialize_unit_2d_dict(self) -> dict:
        xoForce2b = dict()
        for iUnit in self.RadUnit.UnitIdxLst:
            xoForce2b[iUnit] = dict()
        return xoForce2b
    
    #2
    def get_file_extension(self, unitIdxLst: list) -> str:
        extension = ".U"
        tempStr = [str(iUnit+1) for iUnit in sorted(unitIdxLst)]
        return extension + "U".join(tempStr)

    #1
    def get_linking_bonus_2d_array_from_forces(self):
        forceErrorIsolatedSum2 = np.sum((self.XoForceIsolated - self.FullForce)**2)
        # print("Diso", forceErrorIsolatedSum2)
        self.LinkBonus2dArray = self.initialize_linking_bonus_2d_array_with_minimal_distance_matrix()
        for iUnit in self.XoForce2b.keys():
            for jUnit in self.XoForce2b[iUnit].keys():
                forceError2bSum2 = np.sum((self.XoForce2b[iUnit][jUnit] - self.FullForce)**2)
                # print("D2b", iUnit, jUnit, forceError2bSum2)
                value = forceError2bSum2 - forceErrorIsolatedSum2
                if value > self.MINVAL:
                    #TODO abs(value) < self.MINVAL
                    #print("Warning! The linking bonus matrix entry ({0}, {1}) is extermely small.".format(iUnit, jUnit))
                    print("Warning! The linking bonus matrix entry ({0}, {1}) is too negatively small or positive."\
                          .format(iUnit, jUnit))
                    # sys.exit(1)
                else:
                    self.LinkBonus2dArray[iUnit][jUnit] = value
    
    #2
    def initialize_linking_bonus_2d_array_with_minimal_distance_matrix(self) -> dict:
        tmpArray = np.copy(self.RadUnit.MinDist2dArray)
        tmpArray[tmpArray==0] = 0.1
        LinkBonus2dArray = - self.MINVAL / tmpArray
        return LinkBonus2dArray
        
    #2
    # def get_egg_index_set_dict_after_merge(self, unitIdxLst: list) -> dict:
    #     extension = self.get_file_extension(unitIdxLst)
    #     newUnitFile = self.UnitFile + extension
    #     tmpRadiusUnit = RadiusUnit(self.GjfFile, newUnitFile, self.Radius)
    #     tmpRadiusUnit.merge_units(unitIdxLst)
    #     tmpRadiusUnit.write_unit_file(newUnitFile)
    #     return tmpRadiusUnit.InitialRawEggUnitIdxSets
    
    #0
    def write_linking_bonus_2d_array(self, outMatFile: str):
        with open(outMatFile, "w") as wfl:
            for linkBonusArray in self.LinkBonus2dArray:
                for linkBonus in linkBonusArray:
                    print("{0:16.12e}".format(linkBonus), end=" ", file=wfl)
                print(file=wfl)
    
