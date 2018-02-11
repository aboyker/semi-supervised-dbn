# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:10:23 2018

@author: Alexandre Boyker
"""

import tensorflow as tf

def reset_graph():
    
    if 'sess' in globals() and sess:
        sess.close()
        
    tf.reset_default_graph()