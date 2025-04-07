import os
import re
import sys
import time
import numpy as np
from copy import deepcopy
from file.xo import read_data_from_config_file
from lib.misc import get_method_level_from_unit_or_work_path
from lib.misc import print_error_and_exit
from file.other import print_xo_log
from file.submit import SubmitJobWriter
from file.mopac import write_fragment_mop_file
from file.mopac import read_mopac_force, read_mopac_energy, read_mopac_hessian
from file.mopac import is_mopac_calculation_exit_normally_for_a_type
from file.gaussian import write_fragment_gjf_file
from file.gaussian import read_gaussian_force, read_gaussian_energy, read_gaussian_nmr
from file.gaussian import read_gaussian_hessian_from_fchk_file
from file.gaussian import is_gaussian_calculation_exit_normally_for_a_type
from file.orca import read_orca_energy, read_orca_g_tensor, read_orca_scaled_a_tensor, read_orca_superexchange
from file.orca import write_fragment_orca_input_file
from file.orca import is_orca_calculation_exit_normally_for_a_type
from file.xtb import read_xtb_energy 
from file.xtb import write_fragment_xtb_files
from file.qchem import read_qchem_force, read_qchem_energy, write_fragment_qchem_input_file
from file.qchem import is_qchem_calculation_exit_normally_for_a_type
from file.sum import get_result_type_from_input_file
from file.other import get_random_dir_name


ISXN01 = True
TEMPDIRPREF = "/scratch/scr/$USER/xotemp"


class QMManager():
    def __init__(self, configFile: str, inputFile: str, queueName="single", nProc="1", \
                 isMozyme=False, resultType=None):
        # constants #
        # self.FragmentGjfFile = "fragment.gjf"
        # self.FragmentMopFile = "fragment.mop"
        # self.FragmentLogFile = "fragment.log"
        self.FragmentFilePref = "fragment"
        self.FullFilePref = "full"
        self.JobIdFile = "running_jobid"
        self.SubFile = "sub"
        self.ValidResultTypeLst = ["energy", "force", "nmr"]
        # self.DefaultExecutable = "g16"
        # self.ValidExecutableLst = ["g03", "g09", "g16", "mopac"]
        # inputs
        # XoGeneral.__init__()
        self.ConfigFile = configFile
        self.IsMozyme = isMozyme
        self.ResultType = self.get_result_type(inputFile, resultType)
        # derived
        self.Methods, self.ExecutableCommands, self.Executables, self.GaussianExtraInputs \
                = read_data_from_config_file(configFile)
        self.MethodLevelLst = sorted(self.Methods.keys())
        self.ProgramTypes = self.get_method_level_to_program_type_dict()
        self.InputExtentions = self.get_method_level_to_input_extention_dict()
        self.OutputExtentions = self.get_method_level_to_program_output_extension()
        self.CalcTypeKeyWordPerProgramType = self.get_calculate_type_keyword_dict_per_program_dict()
        self.SubPathFileLst = []
        self.OutPathFileLst = []
        self.JobWriter = SubmitJobWriter(queueName)
        if queueName == "local":
            self.Nproc = nProc
        else:
            self.Nproc = str(self.JobWriter.QueueName2Cpu[queueName])

    #0
    def get_result_type(self, inputFile: str, resultType: str) -> str:
        if resultType == None:
            return get_result_type_from_input_file(inputFile)
        elif resultType not in self.ValidResultTypeLst:
            print_error_and_exit("Invalid result type =", resultType)
        else:
            return resultType
    
    #0
    def get_method_level_to_program_type_dict(self) -> dict:
        self.Executable2ProgramType = self.get_executable_to_program_type_dict()
        methodLevel2ProgramType = self.get_method_level_dict(self.Executable2ProgramType)
        return methodLevel2ProgramType

    #1
    def get_executable_to_program_type_dict(self) -> dict:
        executable2ProgramType = dict()
        executable2ProgramType["g03"] = "gaussian"
        executable2ProgramType["g09"] = "gaussian"
        executable2ProgramType["g16"] = "gaussian"
        executable2ProgramType["mopac"] = "mopac"
        executable2ProgramType["xtb"] = "xtb"
        executable2ProgramType["orca"] = "orca"
        executable2ProgramType["qchem"] = "qchem"
        return executable2ProgramType
    
    #1
    def get_method_level_dict(self, executableDict: dict) -> dict:
        methodLevelDict = dict()
        for uLevel in self.Executables.keys():
            if self.Executables[uLevel] in executableDict.keys():
                methodLevelDict[uLevel] = executableDict[self.Executables[uLevel]]
            else:
                print(" Fatal error! Unknown QM program type for the executable =", self.Executables[uLevel], 
                      "defined in configfile =", self.ConfigFile)
                sys.exit(1)
        return methodLevelDict
    
    #0
    def get_method_level_to_program_output_extension(self) -> dict:
        self.Executable2OutputExtention = self.get_executable_to_output_extension_dict()
        methodLevel2OutputExtention = self.get_method_level_dict(self.Executable2OutputExtention)
        return methodLevel2OutputExtention
    
    #1
    def get_executable_to_output_extension_dict(self) -> dict:
        executable2OutputExtention = dict()
        executable2OutputExtention["g03"] = ".log"
        executable2OutputExtention["g09"] = ".log"
        executable2OutputExtention["g16"] = ".log"
        executable2OutputExtention["mopac"] = ".out"
        executable2OutputExtention["xtb"] = ".out"
        executable2OutputExtention["orca"] = "_property.txt" #oout
        executable2OutputExtention["qchem"] = ".out" #oout
        return executable2OutputExtention
    
    #0
    def get_method_level_to_input_extention_dict(self) -> dict:
        self.Executable2InputExtention = self.get_executable_to_input_extension_dict()
        methodLevel2InputExtention = self.get_method_level_dict(self.Executable2InputExtention)
        return methodLevel2InputExtention
        
    #1
    def get_executable_to_input_extension_dict(self) -> dict:
        executable2InputExtention = dict()
        executable2InputExtention["g03"] = ".gjf"
        executable2InputExtention["g09"] = ".gjf"
        executable2InputExtention["g16"] = ".gjf"
        executable2InputExtention["mopac"] = ".mop"
        executable2InputExtention["xtb"] = ".coord"
        executable2InputExtention["orca"] = ".inp"
        executable2InputExtention["qchem"] = ".in"
        return executable2InputExtention
    
    #0
    def get_calculate_type_keyword_dict_per_program_dict(self) -> dict:
        calcTypeKeyWordPerProgramType = dict()
        calcTypeKeyWordPerProgramType["gaussian"] = dict()
        calcTypeKeyWordPerProgramType["mopac"] = dict()
        calcTypeKeyWordPerProgramType["xtb"] = dict()
        calcTypeKeyWordPerProgramType["orca"] = dict()
        calcTypeKeyWordPerProgramType["qchem"] = dict()
        #
        calcTypeKeyWordPerProgramType["gaussian"]["energy"] = ""
        calcTypeKeyWordPerProgramType["gaussian"]["force"] = "force"
        calcTypeKeyWordPerProgramType["gaussian"]["hessian"] = "freq"
        calcTypeKeyWordPerProgramType["gaussian"]["nmr"] = "nmr"
        calcTypeKeyWordPerProgramType["mopac"]["energy"] = ""
        calcTypeKeyWordPerProgramType["mopac"]["force"] = "GRAD OUTPUT(G)"
        calcTypeKeyWordPerProgramType["mopac"]["hessian"] = "DFORCE OUTPUT(F)"
        calcTypeKeyWordPerProgramType["xtb"]["energy"] = ""
        calcTypeKeyWordPerProgramType["orca"]["energy"] = ""
        calcTypeKeyWordPerProgramType["orca"]["tensor"] = ""
        calcTypeKeyWordPerProgramType["qchem"]["energy"] = "sp"
        calcTypeKeyWordPerProgramType["qchem"]["force"] = "force"
        return calcTypeKeyWordPerProgramType
    
    ##### ----- #####
    #0
    def get_temp_file_removal_command_list_at_the_level(self, level: str) -> list:
        cmdLst = []
        if self.ProgramTypes[level] == "orca":
            cmdLst.append("if [ $(ls *.tmp | wc -l) > 0 ];then rm *.tmp;fi")
            cmdLst.append("if [ $(ls *.unso | wc -l) > 0 ];then rm *.unso;fi")
            cmdLst.append("if [ $(ls *.uno | wc -l) > 0 ];then rm *.uno;fi")
            cmdLst.append("if [ $(ls *.qro | wc -l) > 0 ];then rm *.qro;fi")
            cmdLst.append("if [ $(ls *.gori.xyz | wc -l) > 0 ];then rm *.gori.xyz;fi")
            cmdLst.append("if [ $(ls *.gbw | wc -l) > 0 ];then rm *.gbw;fi")
            cmdLst.append("if [ $(ls *.prop | wc -l) > 0 ];then rm *.prop;fi")
        elif self.ProgramTypes[level] == "gaussian":
            pass
        elif self.ProgramTypes[level] == "xtb":
            cmdLst.append("if [ -f xtbtopo.mol ];then rm xtbtopo.mol;fi")
            cmdLst.append("if [ -f wbo ];then rm wbo;fi")
            cmdLst.append("if [ -f xtbrestart ];then rm xtbrestart;fi")
        elif self.ProgramTypes[level] == "mopac":
            pass
        elif self.ProgramTypes[level] == "qchem":
            pass
        else:
            print_error_and_exit("Unsupported program type = {0} at level = {1}".format(\
                                 self.ProgramTypes[level], level))
        return cmdLst
    ##### ----- #####
    
    ##### ----- #####
    #0
    def read_force_function_dict(self) -> dict:
        readFunctions = self.get_read_funcs_per_level({"gaussian":read_gaussian_force, 
                                                       "mopac":read_mopac_force,
                                                       "qchem":read_qchem_force})
        return readFunctions
    
    #1
    def get_read_funcs_per_level(self, funcDict: dict) -> dict:
        readFunctions = dict()
        for level in self.MethodLevelLst:
            if self.ProgramTypes[level] in funcDict.keys():
                readFunctions[level] = funcDict[self.ProgramTypes[level]]
            else:
                print_error_and_exit("Unsupported program type = {0} at level = {1} for read function".format(\
                        self.ProgramTypes[level], level))
        return readFunctions
    
    #0
    def read_energy_function_dict(self) -> dict:
        readFunctions = self.get_read_funcs_per_level({"gaussian":read_gaussian_energy, 
                                                       "mopac":read_mopac_energy, 
                                                       "xtb":read_xtb_energy, 
                                                       "orca":read_orca_energy,
                                                       "qchem":read_qchem_energy})
        return readFunctions
    
    #0
    def read_hessian_function_dict(self) -> dict:
        readFunctions = self.get_read_funcs_per_level({"gaussian":read_gaussian_hessian_from_fchk_file, 
                                                       "mopac":read_mopac_hessian})
        return readFunctions
    
    #0
    def read_nmr_function_dict(self) -> dict:
        readFunctions = self.get_read_funcs_per_level({"gaussian":read_gaussian_nmr})
        return readFunctions
    
    #0
    def read_tensor_dict(self) -> dict:
        readFunctions = self.get_read_funcs_per_level({"orca":[\
                read_orca_g_tensor, read_orca_scaled_a_tensor, read_orca_superexchange]})
        return readFunctions
    ##### ----- #####
    
    ##### ----- #####
    #0
    def get_command_list_from_path(self, uPath: str, fragOrFull="fragment") -> str:
        level = get_method_level_from_unit_or_work_path(uPath)
        inputFile = self.get_input_file_name_from_path(uPath, fragOrFull)
        outFile = self.get_output_file_name_from_path(uPath, fragOrFull)
        cmdLst = []
        if ISXN01:
            xoTempDir = self.get_new_temp_dir()
            cmdLst.append("echo \" Fragment files in dir: `pwd`\"")
            cmdLst.append("if [ ! -d {0} ];then mkdir -p {0};fi".format(xoTempDir))
            cmdLst.append("cp {0} {1}".format(inputFile, xoTempDir))
            cmdLst.append("cd {0}".format(xoTempDir))
            cmdLst.append("echo \" Working in temporary dir: `pwd`\"")
        if self.ProgramTypes[level] == "xtb": 
            cmd = "{0} {1} {2} --parallel {3} > {4}".format(self.ExecutableCommands[level], inputFile, 
                    self.Methods[level], self.JobWriter.QueueName2Cpu, outFile)
        elif self.ProgramTypes[level] == "orca":
            cmd = "{0} {1} > {2}".format(self.ExecutableCommands[level], inputFile, "out.tmp")
        elif self.ProgramTypes[level] == "qchem":
            cmd = "{0} -nt {1} {2} {3}".format(self.ExecutableCommands[level], self.Nproc, inputFile, outFile)
        else:
            cmd = "{0} {1}".format(self.ExecutableCommands[level], inputFile)
        cmdLst.append(cmd)
        cmdLst.extend(self.get_temp_file_removal_command_list_at_the_level(level))
        if ISXN01:
            cmdLst.append("if [ -f *.chk ]; then formchk *.chk; fi")
            cmdLst.append("if [ -f *.fchk ]; then mv *.fchk {0}; fi".format(uPath))
            cmdLst.append("mv {0} {1}".format(outFile, uPath))
            cmdLst.append("cd {0}".format(uPath))
            cmdLst.append("rm {0} -rf".format(xoTempDir))
        return cmdLst
    
    def get_new_temp_dir(self):
        xoTempDir = get_random_dir_name(TEMPDIRPREF)
        while True:
            if os.path.isdir(xoTempDir):
                xoTempDir = get_random_dir_name(TEMPDIRPREF)
            else:
                os.system("mkdir -p " + xoTempDir)
                return xoTempDir
    ##### ----- #####
    
    ##### ----- #####
    #0
    def get_input_file_name_from_path(self, uPath: str, fragOrFull: str) -> str:
        level = get_method_level_from_unit_or_work_path(uPath)
        if fragOrFull == "fragment":
            inputFile =  self.FragmentFilePref + self.InputExtentions[level]
        elif fragOrFull == "full":
            inputFile =  self.FullFilePref + self.InputExtentions[level]
        else:
            print(" Fatal error! Unsupported input = {0} at get_command_list_from_path.".format(fragOrFull))
            sys.exit(1)
        return inputFile
    ##### ----- #####
    
    ##### ----- #####
    #0
    def get_output_file_name_from_path(self, uPath: str, fragOrFull: str) -> str:
        level = get_method_level_from_unit_or_work_path(uPath)
        if fragOrFull == "fragment":
            outputFile =  self.FragmentFilePref + self.OutputExtentions[level]
        elif fragOrFull == "full":
            outputFile =  self.FullFilePref + self.OutputExtentions[level]
        else:
            print(" Fatal error! Unsupported input = {0} at get_command_list_from_path.".format(fragOrFull))
            sys.exit(1)
        return outputFile
    ##### ----- #####
    
    ##### ----- #####
    #0
    def write_fragment_input_file(self, inputFile: str, level: str, nProc: str, mem: str, \
                                  methods: dict, resultType: str, charge: int, spin: int, \
                                  elementLst: list, xyzArray: np.ndarray, bondOrders: dict, \
                                  xyzChargeArray: np.ndarray, basisSets: dict, ecps: dict, \
                                  gaussianExtraInputs: dict, elementSpins=dict(), globalIdxLst=[]):
        calcTypeKeyWord = self.CalcTypeKeyWordPerProgramType[self.ProgramTypes[level]][resultType]
        if calcTypeKeyWord in methods[level]: #todo, this is only for Gaussian and mopac
            calcTypeKeyWord = ""
        if self.ProgramTypes[level] == "gaussian":
            write_fragment_gjf_file(inputFile, nProc, mem, methods[level], calcTypeKeyWord, \
                                    charge, spin, elementLst, xyzArray, bondOrders, \
                                    xyzChargeArray, basisSets[level], ecps[level], gaussianExtraInputs[level])
        elif self.ProgramTypes[level] == "mopac":
            write_fragment_mop_file(inputFile, nProc, methods[level], calcTypeKeyWord, charge, elementLst, \
                                    xyzArray, isMozyme=self.IsMozyme)
        elif self.ProgramTypes[level] == "xtb":
            write_fragment_xtb_files(inputFile, charge, elementLst, xyzArray)
        elif self.ProgramTypes[level] == "orca":
            write_fragment_orca_input_file(inputFile, methods[level], charge, spin, elementLst, xyzArray, \
                                           nProc, mem, elementSpins, globalIdxLst)
        elif self.ProgramTypes[level] == "qchem":
            write_fragment_qchem_input_file(inputFile, methods[level], charge, spin, elementLst, xyzArray, mem,
                    calcTypeKeyWord=calcTypeKeyWord)
        else:
            print_error_and_exit("Invalid program type = {0} at level {1}".format(\
                    self.ProgramTypes[level], level))
    ##### ----- #####
    
    ##### ----- #####
    #0
    def is_calculation_exit_normally_for_a_type(self, path4level: str, outFile: str, resultType: str) -> bool:
        # currently in the workPath
        # outPathFile = os.path.join(workPath, outFile)
        level = get_method_level_from_unit_or_work_path(path4level)
        if self.ProgramTypes[level] == "gaussian":
            return is_gaussian_calculation_exit_normally_for_a_type(outFile, resultType)
        elif self.ProgramTypes[level] == "mopac":
            return is_mopac_calculation_exit_normally_for_a_type(outFile, resultType)
        elif self.ProgramTypes[level] == "orca":
            return is_orca_calculation_exit_normally_for_a_type(outFile, resultType)
        elif self.ProgramTypes[level] == "qchem":
            return is_qchem_calculation_exit_normally_for_a_type(outFile, resultType)
        else:
            print_error_and_exit("Unsupported program type = {0} at level = {1}".format(\
                    self.ProgramTypes[level], level))
    
    #0
    # def is_calculation_exit_normally(self, workPath: str, outFile: str):
    #     outPathFile = os.path.join(workPath, outFile)
    #     if workPath.strip(os.path.sep) != ".":
    #         level = get_method_level_from_unit_or_work_path(workPath)
    #         if self.ProgramTypes[level] == "gaussian":
    #             return is_gaussian_calculation_exit_normally(outPathFile)
    #         elif self.ProgramTypes[level] == "mopac":
    #             return is_mopac_calculation_exit_normally(outPathFile)
    #         elif self.ProgramTypes[level] == "xtb":
    #             return is_xtb_calculation_exit_normally(outPathFile)
    #         else:
    #             print_error_and_exit("Unsupported program type = {0} at level = {1}".format(\
    #                     self.ProgramTypes[level], level))
    #     else:
    #         return is_gaussian_calculation_exit_normally(outPathFile)
    ##### ----- #####

    ##### ----- #####
    #0
    #old submit_jobs_and_write_job_id_to_file
    def submit_jobs_and_write_running_job_id_to_file(self):
        jobIdLst = []
        for subFile in self.SubPathFileLst:
            read = os.popen("bsub < " + subFile).read()
            # Job <96654> is submitted to queue <single>
            match = re.match("Job <([0-9]*)> is submitted to queue", read.strip())
            if match:
                jobIdLst.append(match.group(1))
        self.write_running_job_id_file(jobIdLst)
    
    #1
    def write_running_job_id_file(self, jobIdLst: str):
        with open(self.JobIdFile, "w") as wfl:
            print(" ".join(jobIdLst), file=wfl)
    ##### ----- #####
    
    ##### ----- #####
    #0
    def wait_till_all_jobs_normally_exit(self) -> None:
        MAXCYCLE = 100000
        SLEEPTIME = 10
        SMALLFILE = 20
        iCycle = 0
        nOutFile = len(self.OutPathFileLst)
        while iCycle < MAXCYCLE:
            time.sleep(SLEEPTIME)
            unfinishedPathFileLst = self.get_unfinished_path_file_list(self.OutPathFileLst)
            print_xo_log("  - {0} / {1} files to be finished.".format(len(unfinishedPathFileLst), nOutFile))
            if len(unfinishedPathFileLst) == 0:
                return
            elif len(unfinishedPathFileLst) <= SMALLFILE:
                self.print_unfinished_dir(unfinishedPathFileLst)
            iCycle += 1
        if iCycle >= MAXCYCLE:
            print_error_and_exit("Fragment calculation is not finished within", SLEEPTIME*MAXCYCLE, "s. Exit!")
    
    #1
    def get_unfinished_path_file_list(self, outPathFileLst: list) -> list:
        unfinishedPathFileLst = deepcopy(outPathFileLst)
        for outPathFile in outPathFileLst:
            workPath = os.path.dirname(outPathFile)
            if self.is_calculation_exit_normally_for_a_type(workPath, outPathFile, self.ResultType):
                unfinishedPathFileLst.remove(outPathFile)
        return unfinishedPathFileLst
    
    #1
    def print_unfinished_dir(self, unfinishedPathFileLst: list):
        print_xo_log("  ---------------------- list of unfinished files ---------------------")
        for uPathFile in unfinishedPathFileLst:
            print_xo_log("   " + uPathFile)
        print_xo_log("  ---------------------------------------------------------------------")
    ##### ----- #####
    
    ##### ----- #####
    #0
    def write_sub_file_and_collect_path_file_list(self, pathLst: list, fragOrFull="fragment", nJob=1):
        subPathFileLst, outPathFileLst = self.write_sub_files(pathLst, fragOrFull, nJob)
        self.SubPathFileLst.extend(subPathFileLst)
        self.OutPathFileLst.extend(outPathFileLst)
    
    #1
    def write_sub_files(self, pathLst: list, fragOrFull: str, nJob: int) -> list:
        if nJob >= 1:
            subPathFileLst, outPathFileLst = self.write_several_sub_files(pathLst, fragOrFull, nJob)
        else:
            subPathFileLst, outPathFileLst = self.write_all_sub_files(pathLst, fragOrFull)
        return subPathFileLst, outPathFileLst
    
    #2
    def write_several_sub_files(self, workPathLst: list, fragOrFull: str, nJob=1) -> tuple:
        outPathFileLst = self.get_out_path_file_list(workPathLst, fragOrFull)
        subPathFileLst = [self.SubFile + str(iJob+1) for iJob in range(nJob)]
        cmdLsts = self.get_command_list_for_each_sub_file_successively(workPathLst, fragOrFull, nJob)
        for iJob in range(nJob):
            self.JobWriter.write(subPathFileLst[iJob], cmdLsts[iJob], subPathFileLst[iJob])
        return subPathFileLst, outPathFileLst
    
    #3
    def get_out_path_file_list(self, workPathLst: list, fragOrFull: str) -> list:
        outPathFileLst = []
        for uWorkPath in workPathLst:
            outFile = self.get_output_file_name_from_path(uWorkPath, fragOrFull)
            outPathFileLst.append(os.path.join(uWorkPath, outFile))
        return outPathFileLst
    
    #3
    def get_command_list_for_each_sub_file_successively(self, workPathLst: list, fragOrFull: str, \
                                                        nJob: int) -> dict:
        rootDir = os.getcwd()
        iPath = 0
        cmdLsts = self.initialize_command_list_dict(nJob)
        for iPath, uWorkPath in enumerate(workPathLst):
            level = get_method_level_from_unit_or_work_path(uWorkPath)
            outFile = self.get_output_file_name_from_path(uWorkPath, fragOrFull)
            if not self.is_calculation_exit_normally_for_a_type(uWorkPath, os.path.join(uWorkPath, outFile), 
                                                                self.ResultType):
                iJob = iPath % nJob
                cmdLsts[iJob].append("cd {0}".format(os.path.join(rootDir, uWorkPath)))
                cmdLsts[iJob].extend(self.get_command_list_from_path(os.path.join(rootDir, uWorkPath), fragOrFull))
                iPath += 1
        return cmdLsts
    
    #4
    def initialize_command_list_dict(self, nJob: int) -> dict:
        cmdLsts = dict()
        for iJob in range(nJob):
            cmdLsts[iJob] = []
        return cmdLsts
    
    #2
    def write_all_sub_files(self):
        pass
    ##### ----- #####

