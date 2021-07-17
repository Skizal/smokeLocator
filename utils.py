#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 22:26:11 2021

@author: enrique
"""

from dataclasses import dataclass
from typing import List

@dataclass
class Point:
     x: float
     y: float

@dataclass
class ImageData:
    name: str
    pointA: Point
    pointB: Point
    data: List[int]
    
    