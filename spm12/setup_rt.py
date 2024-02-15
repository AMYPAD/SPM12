'''
All setup for standalone SPM12 using MATLAB Runtime
'''
__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2023"



import os
import platform
from pathlib import Path, PurePath
import requests
import zipfile
import subprocess

from miutil import create_dir

import logging
log = logging.getLogger(__name__)

#----------------------------------------------------------
# > MATLAB standalone:
mat_core = 'https://ssd.mathworks.com/supportfiles/downloads/'
mwin = mat_core+'R2019b/Release/9/deployment_files/installer/complete/win64/MATLAB_Runtime_R2019b_Update_9_win64.zip'
mlnx = mat_core+'R2022b/Release/7/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2022b_Update_7_glnxa64.zip'

# > SPM12 stand-alone core address:
spm_core = 'https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/spm12/'
swin = spm_core+'spm12_r7771_Windows_R2019b.zip'
slnx = spm_core+'spm12_r7771_Linux_R2022b.zip'
smac = spm_core+'spm12_r7771_macOS_R2022b.zip'

spmsa_fldr_name = '.spmruntime'
#----------------------------------------------------------

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def check_platform():

    if not platform.system() in ['Windows', ]: #'Darwin', 'Linux'
        log.error(
            dedent(
                f'''\
                currently the operating system is not supported: {platform.system()}
                only Windows is supported (for now, Linux and macOS coming soon).'''
            )
        )
        raise SystemError('unknown operating system (OS).')

    if platform.system()=='Windows':
        osid = 1
    elif platform.system()=='Linux':
        osid = 0
    elif platform.system()=='Darwin':
        osid = 2
    else:
        osid = None

    return osid
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def get_user_folder():
    user_folder = os.path.expanduser("~")
    return Path(user_folder)
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def check_matlab_rt():
    ''' check if MATLA Runtime exists and what is its position in the
        PATH variable relative to standard MATLAB installation (if
        exists)
    '''

    # > get the environmental variables (PATH in particular)
    ospath = os.environ['PATH']
    ospath = ospath.split(';')

    # > position of MATLAB Runtime and MATLAB (if any exists)
    matrt_pos = None
    mat_pos = None
    for ip, p in enumerate(ospath):
        if 'MATLAB' in p and 'runtime' in p.lower() and 'v97' in p:
            matrt_pos = ip
        if 'MATLAB' in p and not 'runtime' in p.lower():
            mat_pos = ip

    if not (mat_pos is None or matrt_pos is None) and mat_pos<matrt_pos:
        raise ValueError('MATLAB Runtime Path needs to be above standard MATLAB installation Path.\nChange the environment variable Path accordingly.')

    if matrt_pos is not None:
        return True
    else:
        return False
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def standalone_path():
    ''' get the path to standalone SPM
    '''

    # > user main folder
    usrpth = get_user_folder()

    # > core SPM 12 standalone/runtime path
    spmsa_fldr = usrpth/spmsa_fldr_name

    fspm = spmsa_fldr/'spm12'/'spm12.exe'

    return fspm
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def check_standalone():
    ''' Check if the standalone SPM12 is already installed
        with the correct MATLAB Runtime
    '''

    fspm = standalone_path()

    if fspm.is_file() and check_matlab_rt():
        return True
    else:
        return False
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+



#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def ensure_standalone():
    ''' Ensure the standalone SPM12 is installed
        with the correct MATLAB Runtime
    '''

    if not check_standalone():
        log.warning('MATLAB Runtime for SPM12 is not yet installed on your machine')
        response = input('Do you want to install MATLAB Runtime? [y/n]')
        if response in ['y', 'Y', 'yes']:
            install_standalone()
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def get_file(url, save_path):
    response = requests.get(url)

    print('Downloading setup file - this make take a while (it is Matlab) ...')
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f'Successfully downloaded: {save_path}')
    else:
        print(f'Failed to download file. Status code: {response.status_code}')
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+



#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# INSTALL MATLAB RUNTIME and STANDALONE SPM12
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def install_standalone():
    ''' Install Matlab Runtime and the associated SPM12
    '''

    # > select the OS (0: Linux, 1: Windows, 2: MacOS)
    os_sel = check_platform()
    log.info('you are currently using OS platform: {}'.format(platform.system()))

    if os_sel!=1:
        log.error('the operating system is yet supported')
        raise SystemError('OS not supported')
    
    #--------------------------------------------
    # > user main folder
    usrpth = get_user_folder()
    # > core destination path
    spmsa_fldr = usrpth/spmsa_fldr_name
    create_dir(spmsa_fldr)
    # > downloads destination
    dpth = spmsa_fldr/'downloads'
    create_dir(dpth)
    #--------------------------------------------

    if os_sel==1:
        #--------------------------------------------
        if not check_matlab_rt():
            # MATLAB Runtime Installation
            fmwin = dpth/os.path.basename(mwin)
            if not fmwin.is_file():
                get_file(mwin, fmwin)

            # > unzip to MATLAB runtime setup folder 
            matrun_setup = fmwin.parent/'matlab_runtime'
            unzip_file(fmwin, matrun_setup)


            matrun_sexe = [str(f) for f in matrun_setup.iterdir() if f.name=='setup.exe']
            if len(matrun_sexe)!=1:
                raise FileExistsError('Matlab runtime setup executable does not exists or it is confusing')

            try:
                print('AmyPET:>> Running Matlab Runtime Installation - \
                    please approve by pressing yes for administrative privileges')
                subprocess.run(['powershell',  'Start-Process',  matrun_sexe[0], '-Verb', 'Runas'], check=True)
                print("Setup started successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

            if not check_standalone():
                log.warning('MATLAB Runtime not yet installed ...')

        else:
            log.info('MATLAB Runtime already installed')


        #--------------------------------------------

        #--------------------------------------------
        # SPM12
        # > check if SPM12 is already installed
        if not check_standalone():
            fswin = dpth/os.path.basename(swin)
            if not fswin.is_file():
                get_file(swin, fswin)

            unzip_file(fswin, spmsa_fldr)
        else:
            log.info('SPM12 standalone already installed')

        fspm = standalone_path()
       
        if not fspm.is_file():
            raise FileExistsError(
                'The SPM12 executable has not been installed or is missing')
        #--------------------------------------------

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

