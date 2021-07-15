#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:38:13 2021

@author: enrique
"""

import xml.etree.ElementTree as xml

file = xml.parse( "smoke.xml" )
root = file.getroot()


element = root.findall( "bndbox" )
coords = list( element )
for coord in coords:
    print( "Child", child.tag )
