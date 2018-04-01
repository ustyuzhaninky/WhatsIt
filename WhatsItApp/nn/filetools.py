# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:22:48 2017

A set of tools for working with different data files. Time to time it will be 
excented and updated with new features.

@author: Ustyuzhanin K. Yu
"""

import os
import sys
import types
from glob import glob
import yaml
import json
import subprocess
from subprocess import Popen
#from yaml import YAML
#from yaml.compat import StringIO

import pandas

def esave_frame(path, frame, sheet_name='Sheet'):
    '''
    Saves a pandas Data Frame to Excel by given path and checkes if the folder 
    exist or not.
    - Uses xlsxwriter backend
    
    Returns:
     - no output
    
    Arguments:
        - path:
            A directory/file path to save the frame in;
        - frame:
            A pandas data frame to save;
    
    Restrictions:
        path must contain at least one folder to write in, so the function
        cannot write to the script root path.
    '''
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    writer = pandas.ExcelWriter(path,engine='xlsxwriter')
    frame.to_excel(writer,sheet_name)
    writer.save()

def extract_files(path):
    '''
    Extrects the names and root paths of all the files in the given path.
    
    Returns:
        - A list of strings.
    
    Arguments:
        - path:
            The root path to files. Must exist, elsewere the function returns
            an empty list.
    '''
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        return []
    else:
        return [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.*'))]

#print(extract_files('C:\\Users\\Ustyuzhanin K. Yu\\Videos\\GOT\\Season 01'))

def open_frame(path):
    '''
    Opens the frame and checks if the path is not empty.
    
    Returns:
        - pandas dataframe or None
    
    Arguments:
        - path:
            The root path to dataframe file (excel only by now). 
            Must exist, elsewere the function returns None.
    '''
    if not os.path.exists(path):
        return None
    else:
        return pandas.read_excel(path, sheet_name='0')

def yaml_save(yaml_string, filepath):
    '''
    Saves yaml_string to path with enshuring of existance of target dir and file.
    In addition, it estimates current script filepath and uses it to save the file.
    
    '''
    
    #scriptpath = os.path.dirname(__file__)
    #fullname = os.path.join(scriptpath, filepath)
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    #if not os.path.exists(fullname):
    #    open(path).close()
    with open(filepath, 'a+') as stream:
        stream.write(yaml_string)

def crfile(filepath):
    '''
    Creates file in following filepath with all the dirs inside.
    
    '''
    
    #scriptpath = os.path.dirname(__file__)
    #fullname = os.path.join(scriptpath, filepath)
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    open(filepath, 'ab+').close()

def crfileTxt(filepath):
    '''
    Creates file in following filepath with all the dirs inside.
    
    '''
    
    #scriptpath = os.path.dirname(__file__)
    #fullname = os.path.join(scriptpath, filepath)
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    open(filepath, 'at+').close()
    
def yaml_dump(instance, filepath):
    '''
    Dumps any object to yaml file.
    
    '''
    
    #crfile(filepath)
    with open(filepath, 'w+') as yaml_file:
        yaml.dump(instance, yaml_file, default_flow_style=False)
    
def yaml_restore(filepath):#TODO: test for saving complex objects
    '''
    Restores any object from yaml file and returns it.
    May not work with complex ones like classes, functions, class instances, etc.
    
    '''
    
    isntance = None
    with open(filepath, 'r') as yaml_file:
        yaml_string = yaml_file.read()
        isntance = yaml.load(yaml_string)
    return isntance

def yaml_load(yaml_string):
    '''
    Restores any object from given yaml string and returns it.
    May not work with complex ones like classes, functions, class instances, etc.
    
    '''
    
    isntance = yaml.load(yaml_string)
    return isntance

def popExcel(path, isVisible=True):
    '''
    Opens Excel application with file instance and displays it if isVisible is 
    set to "True". To work Excel must be installed and win32com.client.Dispatch 
    is needed to be available.
    Returns a workbook instance.
    
    '''
    
    from win32com.client import Dispatch
    xl = Dispatch('Excel.Application')
    wb = xl.Workbooks.Open('C:\\Documents and Settings\\GradeBook.xls')
    xl.Visible = isVisible    # optional: if you want to see the spreadsheet
    return wb    

def sysStart(string):
    '''
    Launches system command in "string" in system tray (may console occur).
    Do not use with path containing spaceses!!!!
    
    '''
    os.system(string)

def startFile(path):
    '''
    Starts file with system call of standart app for file instance.
    Do not use with path containing spaceses!!!!
    
    '''
    
    os.startfile(path)

def popSubporcess(isWait=False):#TODO: Finish popSubporcess
    '''
    Pushes a subporcess with returning a process pid and instance of 
    Popen in a list.
    
    '''
    
    if not isWait:
        process_one = subprocess.Popen(['gqview', '/home/toto/my_images'])
        return [process_one.pid, process_one]
    else:
        process_one

def popSubprocessExe(filepath):
    '''
    Calls the given exe file in any possible filepath or trows exception.
    
    '''
    subprocess.call([filepath])

def popSubprocessScript(filepath, isWait=True):
    '''
    Calls the given pyhton script in any possible filepath or trows exception.
    Waits for process to end.
    
    '''
    
    os.chdir(os.path.dirname(filepath))
    #print(os.path.basename(filepath), 'python ' + os.path.basename(filepath) )
    pop = Popen('python ' + os.path.basename(filepath))
    if isWait:
        pop.wait()
    #subprocess.call(["python", filepath])
    
def get_mod(modulePath):
    try:
        aMod = sys.modules[modulePath]
        if not isinstance(aMod, types.ModuleType):
            raise KeyError
    except KeyError:
        # The last [''] is very important!
        aMod = __import__(modulePath, globals(), locals(), [''])
        sys.modules[modulePath] = aMod
    return aMod

def get_func(fullFuncName):
    """Retrieve a function object from a full dotted-package name."""

    # Parse out the path, module, and function
    lastDot = fullFuncName.rfind(u".")
    funcName = fullFuncName[lastDot + 1:]
    modPath = fullFuncName[:lastDot]

    aMod = get_mod(modPath)
    aFunc = getattr(aMod, funcName)

    # Assert that the function is a *callable* attribute.
    assert callable(aFunc), u"%s is not callable." % fullFuncName
        # Return a reference to the function itself,
    # not the results of the function.
    return aFunc

def get_class(fullClassName, parentClass=None):
    """Load a module and retrieve a class (NOT an instance).

    If the parentClass is supplied, className must be of parentClass
    or a subclass of parentClass (or None is returned).
    """
    aClass = get_func(fullClassName)

    # Assert that the class is a subclass of parentClass.
    if parentClass is not None:
        if not issubclass(aClass, parentClass):
            raise TypeError(u"%s is not a subclass of %s" %
                            (fullClassName, parentClass))

    # Return a reference to the class itself, not an instantiated object.
    return aClass
