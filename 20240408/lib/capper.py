#!/share/apps/Python/Anaconda3/bin/python3

import sys
import numpy as np
from copy import deepcopy
from file.gaussian import read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file
from file.gaussian import ensure_consistency_between_unit_file_and_gjf_file
from file.other import read_simple_XO2_fragment_set_and_level_dict_from_scheme_file
from file.unit import read_unit_file
from file.other import read_atom_name_list_and_charge_array_from_charge_file
from lib.misc import print_error_and_exit
from file.unit import read_interger_lists_from_file


class UnitFragmentCapper():
    def __init__(self, fullGjfFile: str, unitFile: str, backChargeRange=0.0, backChargeFile="backChargeFile"):
        # input properties
        self.FullGjfFile = fullGjfFile
        self.UnitFile = unitFile
        self.BackChargeRange = backChargeRange
        self.BackChargeFile = backChargeFile
        # constant properties
        self.TooCloseDist = 0.9
        self.BondOrder2CapAtomName = dict([[1.0, "H"], [2.0, "O"]])
        self.AdditionalUnitFile = unitFile + ".add"
        # derived properties
        self.Charge, self.Spin, self.XyzArray, self.ElementLst, self.BondOrders = \
                read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(self.FullGjfFile)
        self.ChargeArray = self.get_valid_charge_array()
        self.AtomIdxSetInRangePerAtoms = self.get_atom_index_set_within_range_per_atom()
        self.TotAtom, self.AtomIdxLstOfUnitLst, self.UnitIdxLst, self.UnitChargeLst, self.UnitSpinLst = read_unit_file(unitFile)
        ensure_consistency_between_unit_file_and_gjf_file(unitFile, fullGjfFile)
        self.get_merged_unit_dicts_from_additional_unit_file()
    
    #0
    def get_merged_unit_dicts_from_additional_unit_file(self):
        self.MergedAtomIdxLstOfUnitLst = deepcopy(self.AtomIdxLstOfUnitLst)
        self.MergedUnitIdxLst = deepcopy(self.UnitIdxLst)
        self.MergedUnitChargeLst = deepcopy(self.UnitChargeLst)
        self.MergedUnitSpinLst = deepcopy(self.UnitSpinLst)
        self.MergedUnitIdxLstInLst = read_interger_lists_from_file(self.AdditionalUnitFile)
        newAtomIdxLstOfUnitLst, newUnitCharge, newUnitSpin = [], 0, 0
        nUnit = len(self.UnitIdxLst)
        for iMergedUnit, unitIdxLst in enumerate(self.MergedUnitIdxLstInLst):
            for iUnit in unitIdxLst:
                #print("D", iUnit, "-", self.MergedAtomIdxLstOfUnitLst[iUnit])
                newAtomIdxLstOfUnitLst += self.MergedAtomIdxLstOfUnitLst[iUnit]
                newUnitCharge += self.MergedUnitChargeLst[iUnit]
                newUnitSpin += self.MergedUnitSpinLst[iUnit] - 1
            self.MergedAtomIdxLstOfUnitLst.append(sorted(newAtomIdxLstOfUnitLst))
            self.MergedUnitIdxLst.append(nUnit + iMergedUnit)
            self.MergedUnitChargeLst.append(newUnitCharge)
            self.MergedUnitSpinLst.append(newUnitSpin + 1)
    
    #0
    def get_valid_charge_array(self) -> np.ndarray:
        if self.BackChargeRange > 0.0:
            atomNameLst, chargeArray = read_atom_name_list_and_charge_array_from_charge_file(self.BackChargeFile)
            self.ensure_consistency_between_charge_file_and_gjf_file(atomNameLst, self.ElementLst)
            return chargeArray
        else:
            return np.array([])
    
    #1
    def ensure_consistency_between_charge_file_and_gjf_file(self, atomNameLst: list, elementLst: list) -> None:
        if len(atomNameLst) != len(elementLst):
            print(" Fatal error! Charge file ({0:d}) and GJF file ({1:d}) contain different numbers of atoms.".\
                    format(len(atomNameLst), len(elementLst)))
            sys.exit(1)
        for uAtom, uElem in zip(atomNameLst, elementLst):
            if uAtom[0].upper() != uElem[0].upper(): #todo
                print(" Fatal error! Inconsistent atom name in Gjf file ({0:s}) and charge file ({1:s})".\
                        format(uAtom, uElem))
    
    #0
    def cap_fragment_from_unit_index_list(self, selUnitIdxLst: list) -> tuple:
        fragCharge = self.get_fragment_charge(selUnitIdxLst)
        fragSpin = self.get_fragment_spin(selUnitIdxLst)
        self.cap_fragment_and_get_new_xyz_element_bondorder_charge_index_from_unit_list(selUnitIdxLst)
        return fragCharge, fragSpin, self.FragXyzArray, self.FragElementLst, self.FragBondOrders, \
                self.FragXyzChargeArray, self.FragAtomIdxLst
    
    #1
    def get_fragment_charge(self, selUnitIdxLst: list) -> int:
        fragCharge = 0
        for iUnit in selUnitIdxLst:
            fragCharge += self.MergedUnitChargeLst[iUnit]
        return fragCharge
        
    #1
    def get_fragment_spin(self, selUnitIdxLst: list) -> int:
        tmp = 0
        for iUnit in selUnitIdxLst:
            tmp += self.MergedUnitSpinLst[iUnit] - 1
        return tmp + 1
    
    #1
    def cap_fragment_and_get_new_xyz_element_bondorder_charge_index_from_unit_list(self, selUnitIdxLst: list):
        self.FragAtomIdxLst = self.get_fragment_atom_index_list_from_unit_index_list(selUnitIdxLst)
        self.get_all_fragment_properties()
    
    #2
    def get_fragment_atom_index_list_from_unit_index_list(self, selUnitIdxLst: list) -> list:
        fragAtomIdxLst = []
        for iUnit in selUnitIdxLst:
            fragAtomIdxLst.extend(self.MergedAtomIdxLstOfUnitLst[iUnit])
        return sorted(fragAtomIdxLst)
    
    #2
    def get_all_fragment_properties(self):
        self.globalIdx2FragmentIdx = self.get_global_to_fragment_atom_index_dict()
        self.FragXyzArray, self.FragElementLst = self.initialize_fragment_xyz_and_element()
        self.FragBondOrders, self.AtomPairLstToCap = \
                self.initialize_fragment_bondorder_and_get_global_atom_index_pair_for_capping_atoms()
        self.FragXyzChargeArray = self.get_background_xyz_charge_array()
        self.append_fragment_xyz_element_bondorder_index_for_capping_atoms()
    
    #3
    def get_global_to_fragment_atom_index_dict(self) -> dict:
        globalIdx2FragmentIdx = dict()
        for iFragAtom, iGlobalAtom in enumerate(self.FragAtomIdxLst):
            globalIdx2FragmentIdx[iGlobalAtom] = iFragAtom
        return globalIdx2FragmentIdx
    
    #3
    def initialize_fragment_xyz_and_element(self) -> tuple:
        fragXyzArray = self.XyzArray[self.FragAtomIdxLst, :]
        fragElementLst = [self.ElementLst[iAtom] for iAtom in self.FragAtomIdxLst]
        return fragXyzArray, fragElementLst
    
    #3
    def initialize_fragment_bondorder_and_get_global_atom_index_pair_for_capping_atoms(self) -> tuple:
        fragBondOrders = dict()
        atomPairLstToCap = []
        for iAtom in self.FragAtomIdxLst:
            iFragAtom = self.globalIdx2FragmentIdx[iAtom]
            fragBondOrders[iFragAtom] = dict()
            for jAtom in self.BondOrders[iAtom].keys():
                if jAtom in self.FragAtomIdxLst:
                    jFragAtom = self.globalIdx2FragmentIdx[jAtom]
                    fragBondOrders[iFragAtom][jFragAtom] = self.BondOrders[iAtom][jAtom]
                else:
                    atomPairLstToCap.append([iAtom, jAtom])
        return fragBondOrders, atomPairLstToCap
    
    #3
    def get_background_xyz_charge_array(self) -> np.ndarray:
        if self.BackChargeRange > 0.0:
            backChargeAtomIdxLst = self.get_background_charge_atom_index_list()
            fragXyzChargeArray = self.get_background_xyz_charge_array_from_atom_index_list(backChargeAtomIdxLst)
            return fragXyzChargeArray
        else:
            return np.array([])
    
    #4
    def get_background_charge_atom_index_list(self) -> list:
        backChargeAtomIdxLst = self.get_atom_index_list_within_selected_range()
        backChargeAtomIdxLst = self.exclude_capped_atom_from_the_list(backChargeAtomIdxLst)
        return backChargeAtomIdxLst
    
    #5
    def get_atom_index_list_within_selected_range(self) -> list:
        atomIdxSetInRange = set([])
        for iAtom in self.FragAtomIdxLst:
            atomIdxSetInRange |= self.AtomIdxSetInRangePerAtoms[iAtom]
        return list(atomIdxSetInRange - set(self.FragAtomIdxLst))
    
    #5
    def exclude_capped_atom_from_the_list(self, backChargeAtomIdxLst: list) -> list:
        for idxPair in self.AtomPairLstToCap:
            if idxPair[1] in backChargeAtomIdxLst:
                 backChargeAtomIdxLst.remove(idxPair[1])
        return backChargeAtomIdxLst

    #4
    def get_background_xyz_charge_array_from_atom_index_list(self, atomIdxLst: list) -> np.ndarray:
        return np.concatenate((self.XyzArray, np.array([self.ChargeArray]).T), axis=1)[atomIdxLst]
    
    #0
    def get_atom_index_set_within_range_per_atom(self) -> dict:
        atomIdxSetInRangePerAtoms = dict()
        cutOff2 = self.BackChargeRange ** 2
        for uIdx, _ in enumerate(self.XyzArray):
            atomIdxSetInRangePerAtoms[uIdx] = set([])
        for uIdx, uXyz in enumerate(self.XyzArray):
            for vIdx, vXyz in enumerate(self.XyzArray):
                if uIdx < vIdx:
                    if np.sum((uXyz - vXyz) ** 2) <= cutOff2:
                        atomIdxSetInRangePerAtoms[uIdx].add(vIdx)
                        atomIdxSetInRangePerAtoms[vIdx].add(uIdx)
        return atomIdxSetInRangePerAtoms
    
    #0
    def append_fragment_xyz_element_bondorder_index_for_capping_atoms_H_only(self): # H
        iAddedFragAtom = len(self.FragElementLst)
        for iAtom, jAtom in self.AtomPairLstToCap:
            iFragAtom = self.globalIdx2FragmentIdx[iAtom]
            capXyz = self.get_cap_atom_xyz(self.XyzArray[iAtom], self.XyzArray[jAtom], self.BondOrders[iAtom][jAtom], \
                    iAtom, jAtom)
            self.FragXyzArray = np.vstack((self.FragXyzArray, capXyz))
            self.FragElementLst.append(self.BondOrder2CapAtomName[self.BondOrders[iAtom][jAtom]])
            self.FragBondOrders[iFragAtom][iAddedFragAtom] = self.BondOrders[iAtom][jAtom]
            self.FragBondOrders[iAddedFragAtom] = dict()
            self.FragBondOrders[iAddedFragAtom][iFragAtom] = self.BondOrders[iAtom][jAtom]
            self.FragAtomIdxLst.append("cap")
            iAddedFragAtom += 1
    
    #3
    def append_fragment_xyz_element_bondorder_index_for_capping_atoms(self): # H + Be
        self.get_xyz_element_bondorder_index_lists_for_capping_atoms()
        self.add_capping_atom_properties_to_the_fragment()
    
    #4
    def get_xyz_element_bondorder_index_lists_for_capping_atoms(self):
        iCapFragAtom = len(self.FragElementLst)
        self.CapXyzLst, self.CapBondOrders, self.CapFragAtomIdxLst, self.CapElementLst = [], dict(), [], []
        for iAtom, jAtom in self.AtomPairLstToCap:
            bondedFragAtomIdx = self.globalIdx2FragmentIdx[iAtom]
            capXyz = self.get_cap_atom_xyz(self.XyzArray[iAtom], self.XyzArray[jAtom], self.BondOrders[iAtom][jAtom], \
                    iAtom, jAtom)
            iList = self.return_very_close_list_index(capXyz, self.CapXyzLst)
            if iList != "":
                closeElement = self.CapElementLst[iList]
                self.CapXyzLst[iList] = self.get_center_xyz(closeElement, self.CapXyzLst[iList], capXyz)
                self.CapBondOrders[self.CapFragAtomIdxLst[iList]][bondedFragAtomIdx] = self.BondOrders[iAtom][jAtom]
                self.CapElementLst[iList] = self.get_center_element(closeElement, self.CapXyzLst[iList])
            else:
                self.CapXyzLst.append(capXyz)
                self.CapBondOrders[iCapFragAtom] = dict()
                self.CapBondOrders[iCapFragAtom][bondedFragAtomIdx] = self.BondOrders[iAtom][jAtom]
                self.CapFragAtomIdxLst.append(iCapFragAtom)
                self.CapElementLst.append(self.BondOrder2CapAtomName[self.BondOrders[iAtom][jAtom]])
                iCapFragAtom += 1
    
    #5
    def get_cap_atom_xyz(self, xyzO: np.ndarray, xyzP: np.ndarray, bondOrder: float, \
            iAtom: int, jAtom: int) -> np.ndarray:
        if bondOrder == 1.0:
            bondLength = 1.0
            return self.normalize_array(xyzP - xyzO) * bondLength + xyzO
        else:
            print("Fatal error! Invalid bondOrder =", bondOrder, "for capping between atom index =", iAtom, "and", jAtom)
            sys.exit(0)
    
    #6
    def normalize_array(self, array: np.ndarray) -> np.ndarray:
        return array / np.sqrt(np.sum(array**2))
    
    #5
    def return_very_close_list_index(self, capXyz: np.ndarray, xyzLst: list) -> int:
        if len(xyzLst) == 0:
            return ""
        for i, uXyz in enumerate(xyzLst):
            if self.calc_array_dist(capXyz, uXyz) <= self.TooCloseDist:
                return i
        return ""
    
    #6
    def calc_array_dist(self, xyz1: np.ndarray, xyz2: np.ndarray) -> float: #todo
        return np.sqrt(np.sum((xyz1 - xyz2)**2))

    #5
    def get_center_xyz(self, closeElement: str, closeXyz: np.ndarray, newXyz: np.ndarray) -> np.ndarray:
        if closeElement == "H":
            weight = 1.0
        elif closeElement == "BE":
            weight = 2.0
        elif closeElement == "B":
            weight = 3.0
        else:
            print_error_and_exit("Invalid capped element =", closeElement, "at xyz =", \
                    closeXyz[0], closeXyz[1], closeXyz[2])
        return (closeXyz * weight + newXyz) / (weight + 1.0)

    #5
    def get_center_element(self, closeElement: str, closeXyz: np.ndarray) -> str:
        if closeElement == "H":
            return "BE"
        elif closeElement == "BE":
            return "B"
        elif closeElement == "B":
            return "C"
        else:
            print_error_and_exit("Invalid capped element =", closeElement, "at xyz =", \
                    closeXyz[0], closeXyz[1], closeXyz[2])
    
    #4
    def add_capping_atom_properties_to_the_fragment(self):
        self.FragElementLst.extend(self.CapElementLst)
        self.FragBondOrders.update(self.CapBondOrders)
        for uXyz, capFragAtomIdx in zip(self.CapXyzLst, self.CapFragAtomIdxLst):
            self.FragXyzArray = np.vstack((self.FragXyzArray, uXyz))
            self.FragAtomIdxLst.append("cap")
            for bondedAtomIdx in self.CapBondOrders[capFragAtomIdx].keys():
                self.FragBondOrders[bondedAtomIdx][capFragAtomIdx] = \
                        self.CapBondOrders[capFragAtomIdx][bondedAtomIdx]


class FragmentCapper(UnitFragmentCapper):
    def __init__(self, schemeFile: str, fullGjfFile: str, backChargeRange=0.0, backChargeFile=""):
        # input properties
        self.SchemeFile = schemeFile
        self.FullGjfFile = fullGjfFile
        self.BackChargeRange = backChargeRange
        self.BackChargeFile = backChargeFile
        # derived properties
        self.Charge, self.Spin, self.XyzArray, self.ElementLst, self.BondOrders = \
                read_charge_spin_coordinates_elements_and_bondorder_from_gjf_file(self.FullGjfFile)
        self.ChargeArray = self.get_valid_charge_array()
        self.AtomIdxSetInRangePerAtoms = self.get_atom_index_set_within_range_per_atom()
        #self.TotAtom = len(self.ElementLst)
        self.EggIdxSets, self.EggLevels, _, _ = read_simple_XO2_fragment_set_and_level_dict_from_scheme_file(schemeFile)
        self.TooCloseDist = 0.9
        self.BondOrder2CapAtomName = dict([[1.0, "H"], [2.0, "O"]])
    
    #####
    def cap_fragment_from_atom_index_list(self, atomIdxLst: list) -> tuple:
        fragCharge = self.get_fragment_charge()
        fragSpin = self.get_fragment_spin()
        self.cap_fragment_and_get_new_xyz_element_bondorder_index_from_atom_index(atomIdxLst)
        return fragCharge, fragSpin, self.FragXyzArray, self.FragElementLst, self.FragBondOrders, \
                self.FragXyzChargeArray, self.FragAtomIdxLst
    
    def get_fragment_charge(self) -> int:
        return 0
    
    def get_fragment_spin(self) -> int:
        return 1
    
    ####
    def cap_fragment_and_get_new_xyz_element_bondorder_index_from_atom_index(self, atomIdxLst: list) -> None:
        self.FragAtomIdxLst = deepcopy(atomIdxLst)
        self.get_all_fragment_properties()
    
    #####
    def cap_fragment_from_fragment_name(self, fragName: str) -> tuple:
        fragCharge = self.get_fragment_charge()
        fragSpin = self.get_fragment_spin()
        self.cap_fragment_and_get_new_xyz_element_bondorder_index_from_fragment_name(fragName)
        return fragCharge, fragSpin, self.FragXyzArray, self.FragElementLst, self.FragBondOrders, self.FragAtomIdxLst
    
    ####
    def cap_fragment_and_get_new_xyz_element_bondorder_index_from_fragment_name(self, fragName: str) -> None:
        self.FragAtomIdxLst = self.get_fragment_atom_index_list_from_fragment_name(fragName)
        self.get_all_fragment_properties()
    
    def get_fragment_atom_index_list_from_fragment_name(self, fragName: str) -> list:
        fragNameLst = fragName.split(".")
        fragAtomIdxSet = self.EggIdxSets[fragNameLst[0]]
        for fragName in fragNameLst[1:]:
            fragAtomIdxSet = fragAtomIdxSet.union(self.EggIdxSets[fragName])
        return sorted(fragAtomIdxSet)


### test with test_class_cap_fragment.ipynb
def main():
    # options
    gjfFile = "2porphyrin-seg.gjf"
    schemeFile = "schemefile"
    fragName = "R2"
    # 
    fragmentToCap = FragmentCapper(schemeFile, gjfFile)
    fragCharge, fragSpin, fragXyzArray, fragElementLst, fragBondOrders, fragAtomIdxLstWithCap \
            = fragmentToCap.cap_fragment_from_atom_index_list(list(fragmentToCap.EggIdxSets[fragName]))
    # output
    print(fragCharge, fragSpin)
    for uXyz, uElement in zip(fragXyzArray, fragElementLst):
        print("{0:3s} {1:15.8f} {2:15.8f} {3:15.8f}".format(uElement, uXyz[0], uXyz[1], uXyz[2]))
    for uKey in fragBondOrders.keys():
        print(uKey, fragBondOrders[uKey])
    print(fragAtomIdxLstWithCap)


if __name__ == "__main__":
    main()

