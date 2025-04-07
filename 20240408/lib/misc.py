import re
import os
import sys
import stat
import shutil
from copy import deepcopy


VALIDMETHODLEVELLST = ["H", "L"]


##
def delete_dir(dirName: str) -> None:
    # from: https://blog.csdn.net/yanlintao1/article/details/80102286
    if os.path.exists(dirName):
        for fileList in os.walk(dirName):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0],name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0],name))
        shutil.rmtree(dirName)
        return "Delete directory " + dirName + " ok."
    else:
        return "No directory " + dirName + " found."

###
def create_dir_if_not_exist(dirName: str):
    if os.path.isfile(dirName):
        os.unlink(dirName)
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

###
def backup_and_create_dir(dirName: str) -> None:
    backup_dir(dirName)
    os.mkdir(dirName)

def backup_dir(dirName: str) -> None:
    if os.path.isdir(dirName):
        backDir = dirName + ".BAK"
        if os.path.isdir(backDir):
            delete_dir(backDir)
        os.rename(dirName, backDir)

##
def backup_file(filename: str) -> None:
    if os.path.isfile(filename):
        backfile = filename + ".BAK"
        if os.path.isfile(backfile):
            os.unlink(backfile)
        os.rename(filename, backfile)

##
def backup_serial_file(fileName: str) -> None:
    i = 0
    MAXI = 1000
    if os.path.isfile(fileName):
        for i in range(1, MAXI + 1):
            backFile = fileName + "." + str(i) + ".BAK"
            if not os.path.isfile(backFile):
                os.rename(fileName, backFile)
                return
        print("Warning! Too may backup files. Increase MAXI =", MAXI, \
              "in backup_serial_file of misc.py if you really need a larger value.")

###
def flatten(l: list):  
    for el in l:  
        if hasattr(el, "__iter__") and not isinstance(el, type(" ")):  
            for sub in flatten(el):  
                yield sub  
        else:  
            yield el  

##### ----- #####
#####
def get_number_index_list_from_serial_strings(numStrings: str, sep=",") -> list:
    # 1, 4-6, 8-10 -> [0, 3, 4, 5, 7, 8, 9]
    numLineLst = numStrings.split(sep)
    numLst = []
    if len(numLineLst) >= 1:
        for u in numLineLst:
            numLst.extend(get_number_index_list_from_serial_string(u))
    return numLst

def get_number_index_list_from_serial_string(numStr: str) -> list:
    # 4-6 -> [3, 4, 5]
    # 4 -> [3]
    i = numStr.find("-")
    try:
        if i > -1:
            return list(range(int(numStr[0:i]) - 1, int(numStr[i+1:])))
        else:
            return [int(numStr) - 1]
    except ValueError:
        print(" Fatal error! Invalid integer string", numStr)
        sys.exit(1)
##### ----- #####

##### ----- #####
#####
def get_number_index_list_from_index_strings(numStrings: str, sep=",") -> list:
    # 0, 4-6, 8-10 -> [0, 4, 5, 6, 8, 9, 10]
    numLineLst = numStrings.split(sep)
    numLst = []
    if len(numLineLst) >= 1:
        for u in numLineLst:
            numLst.extend(get_number_index_list_from_index_string(u))
    return numLst

def get_number_index_list_from_index_string(numStr: str) -> list:
    # 4-6 -> [4, 5, 6]
    # 4 -> [4]
    i = numStr.find("-")
    if i > -1:
        return list(range(int(numStr[0:i]), int(numStr[i+1:]) + 1))
    else:
        return [int(numStr)]
##### ----- #####


def get_subset_index_and_parent_set_index(setLst: list) -> tuple:
    subSetIdxLst, parentSetIdxLst = [], []
    nSet = len(setLst)
    for iSet in range(nSet):
        for jSet in range(iSet+1, nSet):
            if setLst[iSet] <= setLst[jSet]:
                subSetIdxLst.append(iSet)
                parentSetIdxLst.append(jSet)
            elif setLst[iSet] > setLst[jSet]:
                subSetIdxLst.append(jSet)
                parentSetIdxLst.append(iSet)
    return sorted(subSetIdxLst)[::-1], sorted(parentSetIdxLst)[::-1]

# ----- #
###
def get_subset_key_and_parent_set_key_at_different_levels(sets: dict, levels: dict) -> tuple:
    subSetKeyLst, parentSetKeyLst = [], []
    uniqLevelLst = list(set(levels))
    for uLevel in uniqLevelLst:
        tmpSets = get_sets_at_same_level(sets, levels, uLevel)
        newSubSetKeyLst, newParentSetKeyLst = get_subset_key_and_parent_set_key(tmpSets)
        subSetKeyLst.extend(newSubSetKeyLst)
        parentSetKeyLst.extend(newParentSetKeyLst)
    return sorted(subSetKeyLst), sorted(parentSetKeyLst)
        
def get_sets_at_same_level(sets: dict, levels: dict, uLevel: str) -> dict:
    tmpSets = dict()
    for uName in sets.keys():
        if levels[uName] == uLevel:
            tmpSets[uName] = sets[uName]
    return tmpSets
         
def get_subset_key_and_parent_set_key(sets: dict) -> tuple:
    subSetKeyLst, parentSetKeyLst = [], []
    nSet = len(sets)
    setKeyLst = list(sets.keys())
    for iSet in range(nSet):
        iKey = setKeyLst[iSet]
        for jSet in range(iSet+1, nSet):
            jKey = setKeyLst[jSet]
            if sets[iKey] <= sets[jKey]:
                subSetKeyLst.append(iKey)
                parentSetKeyLst.append(jKey)
            elif sets[iKey] > sets[jKey]:
                subSetKeyLst.append(jKey)
                parentSetKeyLst.append(iKey)
    return subSetKeyLst, parentSetKeyLst
# ----- #

##### ----- #####
def ensure_file_exists(fileName: str, fileType: str):
    if not os.path.isfile(fileName):
        print("Fatal error! No", fileType, "=", fileName, "exists.")
        sys.exit(0)
##### ----- #####

##### ----- #####
def compress_continous_index_into_serial_string(idxLst: list, sep="_") -> str:
    if idxLst != "full":
        outxt = ""
        serialLst = sorted([i+1 for i in idxLst])
        iStart = iEnd = serialLst[0]
        for i in serialLst[1:]:
            if i == iEnd + 1:
                iEnd = i
            else:
                outxt = add_to_outxt(outxt, iStart, iEnd, sep)
                iStart = iEnd = i
        outxt = add_to_outxt(outxt, iStart, iEnd, sep)
    else:
        outxt = "full"
    return outxt

def add_to_outxt(outxt: str, iStart: int, iEnd: int, sep: str) -> str:
    if iStart != iEnd:
        if outxt == "":
            outxt = str(iStart) + "-" + str(iEnd)
        else:
            outxt += sep + str(iStart) + "-" + str(iEnd)
    else:
        if outxt == "":
            outxt = str(iStart)
        else:
            outxt += sep + str(iStart)
    return outxt
##### ----- #####

##### ----- #####
# [1,2,3,4,5,43,46,47,78,79,80,81] -> U1-5_43_46-47_78-81
def get_dir_name_from_index_list(dir1stChar: str, idxLst: list) -> str:
    if len(dir1stChar) != 1:
        print("Fatal error! Only one character is allowed to initiate the dir.")
        sys.exit(0)
    return dir1stChar + compress_continous_index_into_serial_string(idxLst)

# U1-5_43_46-47_78-81 -> [1,2,3,4,5,43,46,47,78,79,80,81]
def get_integer_list_from_dir_name(fragmentDirName: str) -> list:
    idxLst = []
    for numStr in fragmentDirName[1:].split("_"):
        idxLst += get_number_index_list_from_index_string(numStr)
    return idxLst
##### ----- #####

##### ----- #####
def mk_serial_dir(pathStr: str) -> None:
    dirLst = pathStr.split(os.path.sep)
    nDir = len(dirLst)
    for iDir in range(nDir):
        if not os.path.isdir(os.path.join(*dirLst[:iDir+1])):
            os.mkdir(os.path.join(*dirLst[:iDir+1]))
##### ----- #####

##### ----- #####
def get_number_of_heavy_atoms_from_element_list(elementLst: list, excludeLst=["H"]) -> int:
    iHeavyAtom = 0
    for element in elementLst:
        if element.upper() not in excludeLst:
            iHeavyAtom += 1
    return iHeavyAtom
##### ----- #####

##### ----- #####
def get_all_index_set_from_dict(eggIdxSets: dict) -> set:
    allIdxSet = set([])
    for idxSet in eggIdxSets.values():
        allIdxSet |= idxSet
    return allIdxSet

def get_full_index_set_from_dict(eggIdxSets: dict) -> set:
    return get_all_index_set_from_dict(eggIdxSets)
##### ----- #####
    
##### ----- #####
def remove_subset_in_dict(idxSets: dict) -> dict:
    noSubSetIdxSets = dict()
    for newName in sorted(idxSets.keys()):
        isAdd = True
        loopDict = deepcopy(noSubSetIdxSets)
        for uName, uSet in loopDict.items():
            if idxSets[newName] <= uSet:
                isAdd = False
                break
            elif idxSets[newName] > uSet:
                del noSubSetIdxSets[uName]
        if isAdd:
            noSubSetIdxSets[newName] = idxSets[newName]
    return noSubSetIdxSets
##### ----- #####

##### ----- #####
def get_non_number_character_list_from_atom_name_list(lineLst: list) -> list:
    charLst = []
    for line in lineLst:
        charLst.append(get_first_non_number_continuous_characters(line))
    return charLst
    
def get_first_non_number_continuous_characters(line: str) -> str:
    search = re.search("(\D+).*", line)
    return search.group(1)
##### ----- #####

##### ----- #####
def get_method_level_and_index_list_from_calc_path(fullPath: str) -> tuple:
    tmp = fullPath.split(os.path.sep)
    level = tmp[-1]
    idxLst = get_index_list_from_dir_name(tmp[-2]) # name is serial
    return level, idxLst

def get_index_list_from_dir_name(fragmentDirName: str) -> list: # name is serial
    idxLst = []
    for numStr in fragmentDirName[1:].split("_"):
        idxLst += get_number_index_list_from_serial_string(numStr)
    return idxLst
##### ----- #####

##### ----- #####
def print_error_and_exit(*args):
    print('Fatal error!', *args)
    sys.exit(0)
##### ----- #####


##### ----- #####
def change_dict_key(oldDict: dict, oldKeyLst: list, newKeyLst: list) -> dict:
    newDict = dict()
    for oldk, newk in zip(oldKeyLst, newKeyLst):
        newDict[newk] = oldDict[oldk]
    return newDict
##### ----- #####


##### ----- #####
def get_method_level_from_unit_or_work_path(path: str) -> str:
    if os.path.sep in path:
        level = path.rstrip("os.path.sep")[-1]
        if level in VALIDMETHODLEVELLST:
            return level
        else:
            return "H"
    else:
        return "H"
##### ----- #####


##### ----- #####
def get_center_unit_index_from_egg_name(eggName: str) -> int:
    # eggName = "U" + str(uid + 1) + "F" + str(iFragment + 1)
    match = re.match("U(\d+)F1", eggName)
    if match:
        return int(match.group(1)) - 1
    else:
        print_error_and_exit("Invalid egg name = {0}".format(eggName))
##### ----- #####


##### ----- #####
def get_mem_in_megabytes_from_gaussian_mem_string(gaussianMemString: str) -> int:
    match = re.match("([\d]+)([GWM]B)", gaussianMemString)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "GB":
            amount *= 1000
        elif unit == "WB":
            amount *= 8
        elif unit == "MB":
            amount *= 1
        else:
            print_error_and_exit("Invalid Gaussian mem string =", gaussianMemString)
    else:
        print_error_and_exit("Invalid Gaussian mem string =", gaussianMemString)
    return amount
##### ----- #####


##### ----- #####
def dict_to_lists_in_list(tempDict: dict) -> list:
    outLst = []
    for i in sorted(tempDict.keys()):
        outLst.append(tempDict[i])
    return outLst

def pair_lists_to_set_dict(pairIdxLst: list) -> dict:
    unitPairSets = dict()
    for [iUnit, jUnit] in pairIdxLst:
        if iUnit > jUnit:
            iUnit, jUnit = jUnit, iUnit
        isInSet = False
        for kUnit in unitPairSets.keys():
            if iUnit in unitPairSets[kUnit] or jUnit in unitPairSets[kUnit]:
                unitPairSets[kUnit].add(iUnit)
                unitPairSets[kUnit].add(jUnit)
                isInSet = True
        if not isInSet:
            unitPairSets[iUnit] = set([iUnit, jUnit])
    return unitPairSets
##### ----- #####


##### ----- #####
def read_uncommented_lines(filename: str):
    with open(filename) as rfl:
        allLineLst = rfl.readlines()
    outLineLst = []
    for line in allLineLst:
        line = line.strip()
        if len(line) == 0 or line[0] != "#":
            outLineLst.append(line)
    return outLineLst
##### ----- #####


