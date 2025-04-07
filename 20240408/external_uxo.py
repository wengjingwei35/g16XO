#!/share/apps/Python/Anaconda3/bin/python3

import os
import sys
from lib.misc import ensure_file_exists, print_error_and_exit
from file.gaussian import read_gaussian_external_input_file
from file.gaussian import read_nproc_and_mem_string_from_gjf_file
from file.gaussian import read_resource_and_route_from_gjf_file
from file.gaussian import read_charge_spin_coordinates_freeze_elements_bondorder_modredundant_from_gjf_file
from file.gaussian import write_fragment_gjf_file_with_lines
from file.gaussian import write_gaussian_external_output_file
from file.gaussian import read_gaussian_energy
from file.gaussian import read_gaussian_force
from file.gaussian import read_gjf_route_line
from lib.xo import UnitBasedXoProcessor
import numpy as np


XOPARAMFILE = "xofile"
BANNEDGJFFILE = "xo.gjf"
NEWGJFFILE = "__tmp__.gjf"
NEWLOGFILE = "__tmp__.log"
# 0=energy only, 1=first derivatives, 2=second derivatives
DERIVATIVELEVEL2ROUTE = {0:"# energy", 1:"# force", 2:"# hessian"}
VALIDQUEUENAMELST = ["local", "Gaussian", "xp28mc6", "xp72mc10", "xp48mc8", "xp40mc12", "single", "small"]
CONFIGFILE = "configfile"


def main():
    if len(sys.argv) != 7 and len(sys.argv) != 1: # error
        print_error_and_exit("Fatal error! Wrong number of parameters ({0}) for external_xo program".format(\
                             len(sys.argv) - 1))
    else:
        check_input_parameter_file()
        if len(sys.argv) == 7: # run
            print_start_information()
            gjfFile, unitFile, eggFile, configFile, queueName, nJob, nProcStr, memStr = read_xo_parameter_file()
            ensure_valid_external_layer_name(sys.argv[1])
            inputExternalFile, outputExternalFile = sys.argv[2], sys.argv[3]
            write_gjf_file_for_xo_and_print_xyz(gjfFile, inputExternalFile, NEWGJFFILE, nProcStr, memStr)
            run_external_xo(outputExternalFile, unitFile, eggFile, queueName, nJob, nProcStr, memStr)
            remove_temp_files()
            print_end_information()

#0
def check_input_parameter_file():
    print("================================================================================")
    print(" Checking whether external XO input files exist ...")
    ensure_file_exists(XOPARAMFILE, "xo parameter file")
    gjfFile, unitFile, eggFile, configFile, queueName, nJob, nCpu, mem = read_xo_parameter_file()
    ensure_valid_gjf_file(gjfFile)
    ensure_file_exists(unitFile, "unit file")
    ensure_file_exists(eggFile, "egg file")
    ensure_file_exists(configFile, "config file")
    if queueName not in VALIDQUEUENAMELST:
        print_error_and_exit("Invalid queue name =", queueName)
    print(" Go ahead! All files for running external XO exist.")
    print("================================================================================")

#1
def read_xo_parameter_file() -> tuple:
    contents = get_default_settings()
    print(" Reading XO parameter file ...")
    with open(XOPARAMFILE) as rfl:
        for line in rfl:
            line = line.strip()
            if len(line) > 0:
                splittedLine = line.split("=")
                if len(splittedLine) == 2:
                    option = splittedLine[0].strip()
                    content = splittedLine[1].strip()
                    if option.lower() in contents.keys():
                        contents[option] = content
                    else:
                        print_correct_xo_parameter_file_format()
                        print_error_and_exit("Invalid option ({0}) for xo parameter file ({1})".format(\
                                             option, XOPARAMFILE))
                else:
                    print_correct_xo_parameter_file_format()
                    print_error_and_exit("Invalid line ({0}) for xo parameter file ({1})".format(\
                                         line, XOPARAMFILE))
    validate_job_number(contents["job number"])
    print(" Reading XO parameter file finished.")
    return contents["gjffile"], contents["unitfile"], contents["eggfile"], \
           contents["configfile"], contents["queue name"], int(contents["job number"]), \
           contents["processor number"], contents["memory"]

#2
def validate_job_number(jobStr: str):
    try:
        _ = int(jobStr)
    except ValueError:
        print_error_and_exit(" Invalid job number =", jobStr)

#2
def get_default_settings() -> dict:
    contents = dict()
    contents["gjffile"] = "system.gjf"
    contents["unitfile"] = "unitfile"
    contents["eggfile"] = "eggFile"
    contents["configfile"] = CONFIGFILE
    contents["queue name"] = "local"
    contents["job number"] = 1
    contents["processor number"] = "1"
    contents["memory"] = "10GB"
    return contents

#2 
def print_correct_xo_parameter_file_format():
    print("------- xo parameter file -------")
    print(" gjffile    = system.gjf")
    print(" unitfile   = unitfile   [optional]")
    print(" eggfile    = eggFile    [optional]")
    print(" configfile = configfile [optional]")
    print(" queue name = local      [optional]")
    print(" job number = 1          [optional]")
    print("----------------------------------")

#1
def ensure_valid_gjf_file(gjfFile: str):
    if gjfFile == BANNEDGJFFILE:
        print()
        print("!" * 100)
        print(" Fatal error! gjfFile =", gjfFile, "is used by XO internal code. Please use other names.")
        print("!" * 100)
        print()
        sys.exit(1)
    ensure_file_exists(gjfFile, "gjf file")
    routeLineLower = read_gjf_route_line(gjfFile).lower()
    if "opt" in routeLineLower and "nomicro" not in routeLineLower:
        print()
        print("!" * 100)
        print("Fatal error! opt(nomicro) MUST be used in gjf file for optimization jobs.")
        print("!" * 100)
        print()
        sys.exit(1)
    if ("opt" in routeLineLower or "force" in routeLineLower) and "nosymm" not in routeLineLower:
        print()
        print("!" * 100)
        print("Fatal error! nosymm MUST be used in gjf file for optimization or force jobs.")
        print("!" * 100)
        print()
        sys.exit(1)

#0
def print_start_information():
    print("================================================================================")
    print("============================= Start XO Calculation =============================")
    print("================================================================================")

#0
def ensure_valid_external_layer_name(layer: str) -> None:
    if layer != "R":
        print_error_and_exit("Fatal error! Invalid layer =", layer)

#0
def write_gjf_file_for_xo_and_print_xyz(gjfFile: str, inputExternalFile: str, newGjfFile: str, nprocStr: str, memStr: str):
    print(" Writing XO input GJF file ...")
    oriMemStr, oriNprocStr, chkStr, oriRouteLine = read_resource_and_route_from_gjf_file(gjfFile)
    charge, spin, oriXyzArray, freezeLst, elementLst, modredundantFreezeInfoLst, bondOrders = \
            read_charge_spin_coordinates_freeze_elements_bondorder_modredundant_from_gjf_file(gjfFile)
    modredundantFreezeInfoLst = []
    extAtom, derivativeLevel, extCharge, extSpin, extXyzArray, _ = \
            read_gaussian_external_input_file(inputExternalFile)
    ensure_consistent_system_information(charge, extCharge, spin, extSpin, oriXyzArray, extXyzArray)
    nprocLine = "%nproc=" + nprocStr
    memLine = "%mem=" + memStr
    routeLine = get_route_line(derivativeLevel)
    write_fragment_gjf_file_with_lines(newGjfFile, memLine, nprocLine, chkStr, routeLine, \
                                       charge, spin, elementLst, extXyzArray, bondOrders, \
                                       freezeLst=freezeLst, modredundantFreezeInfoLst=modredundantFreezeInfoLst)
    print(" Writing XO input GJF file finished.")
    print_xyz(elementLst, extXyzArray)

#1
def ensure_consistent_system_information(charge, extCharge, spin, extSpin, xyzArray, extXyzArray):
    if charge != extCharge or spin != extSpin or xyzArray.shape != extXyzArray.shape:
        print_error_and_exit("Fatal error! Inconsistent gjf file and external input file.")

#1
def get_route_line(derivativeLevel: str) -> str:
    return get_valid_route_line_from_derivative_level(derivativeLevel) + " geom=conn nosymm"

#2
def get_valid_route_line_from_derivative_level(derivativeLevel: str) -> str:
    if derivativeLevel in DERIVATIVELEVEL2ROUTE.keys():
        return DERIVATIVELEVEL2ROUTE[derivativeLevel]
    else:
        print(" Invalid derivative level = {} obtained.".format(derivativeLevel))
        sys.exit(1)

#1
def print_xyz(elementLst: list, xyzArray: np.ndarray):
    print("                          Input orientation:")
    print(" ---------------------------------------------------------------------")
    print(" Center     Atomic      Atomic             Coordinates (Angstroms)")
    print(" Number     Number       Type             X           Y           Z")
    print(" ---------------------------------------------------------------------")
    for iAtom, (element, xyz) in enumerate(zip(elementLst, xyzArray)):
        print("{0:7d}{1:>11s}{2:12d}{3:16.6f}{4:12.6f}{5:12.6f}".format(iAtom+1, element, 0, xyz[0], xyz[1], xyz[2]))
    print(" ---------------------------------------------------------------------")
    """
      1          6           0        0.000000    0.000000    0.000000
      2          1           0        0.000000    0.000000    1.089000
      3          1           0        1.026719    0.000000   -0.363000
      4          1           0       -0.513360   -0.889165   -0.363000
      5          1           0       -0.513360    0.889165   -0.363000
      6          1           0        0.000000    0.000000    3.089000
    """

#0
def run_external_xo(outputExternalFile: str, unitFile: str, eggFile: str, queueName: str, nJob: int, \
                    nProcStr: str, mem: str):
    print(" Runing XO calculation ...")
    # nProc, mem = read_nproc_and_mem_string_from_gjf_file(NEWGJFFILE)
    processor = UnitBasedXoProcessor(NEWGJFFILE, unitFile, eggFile, "subXO", ".unitxo", \
                                     queueName, int(nProcStr), mem, nJob, False, 0.0, "", "", \
                                     calculateFragmentEveryTime=True, useStandardOrientation=False)
    processor.derive_fragments_and_write_weight_path_alias_and_fragment_files(isUnit=True)
    processor.calculate()
    processor.collect_result_list_and_write_gaussian_external_file(outputExternalFile)
    print(" Runing XO calculation finished.")

#0
def remove_temp_files():
    if os.path.isfile(NEWGJFFILE):
        os.unlink(NEWGJFFILE)
    if os.path.isfile(NEWLOGFILE):
        os.unlink(NEWLOGFILE)

#0
def print_end_information():
    print("================================================================================")
    print("============================ XO Calculation Finished ===========================")
    print("================================================================================")


if __name__ == "__main__":
    main()

