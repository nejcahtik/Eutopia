# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:51:23 2021

@author: SNT
"""
#%%############################################################################
'''                      create_working_environment                         '''
###############################################################################

import os


def create_working_environment(working_dir, subdirs = ['trained_models', 'generated_text']):
    if working_dir is None:
        from datetime import datetime
        now = datetime.now() 
        working_dir = now.strftime("%m_%d_%H_%M")
    
    list_dir = os.listdir()
    if not working_dir in list_dir:
        os.mkdir(working_dir)
    
    handled_dirs = {}
    list_dir = os.listdir(working_dir)
    for d in subdirs:
        if not d in list_dir:
            os.mkdir(working_dir+'/'+d)
        handled_dirs[d] = working_dir+'/'+d
    return handled_dirs

#%%############################################################################
'''                       create_syntax_tag_mapper                          '''
###############################################################################


def store_experimental_settings(dict_settings, workdir):
    ouptut_text = ''
    for k in dict_settings.keys():
        ouptut_text += '{} = {}\n'.format(k, dict_settings[k])

    with open(os.path.join(workdir, 'settings.txt'), 'w', encoding = 'utf-8') as f:
        f.write(ouptut_text)
    return 1








