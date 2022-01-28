
# Ashwin Goyal

#%%
# A naive cross-matcher

#1
def hms2dec(h,m,s):
  return 15*(h+m/60 + s/3600)

def dms2dec(d,m,s):
  if d >= 0:
    return d + m/60 + s/3600
  return -(-d + m/60 + s/3600)

#2

import numpy as np

def angular_dist(ra1,dec1,ra2,dec2):
  r1,d1,r2,d2 = np.radians(ra1),np.radians(dec1),np.radians(ra2),np.radians(dec2)
  b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
  a = np.sin(np.abs(d1 - d2)/2)**2
  d = 2*np.arcsin(np.sqrt(a + b))
  return np.degrees(d)

#3



def import_bss():
  cat = np.loadtxt('bss.dat', usecols=range(1, 7))
  l = []
  for i in range(len(cat)):
    l.append(( (i+1,) + 
              (float(hms2dec(cat[i,0],cat[i,1],cat[i,2])),) + 
              (float(dms2dec(cat[i,3],cat[i,4],cat[i,5])),) ))
  return l

def import_super():
  cat = np.loadtxt('super.csv', delimiter=',', skiprows=1, usecols=[0, 1])
  l = []
  for i in range(len(cat)):
    l.append((i+1,) + tuple(cat[i]))
  return l

#4

def find_closest(cat,ra,dec):
  closest = 0
  for i in range(len(cat)):
    if angular_dist(cat[i][1],cat[i][2],ra,dec) < angular_dist(cat[closest][1],cat[closest][2],ra,dec):
      closest = i
  
  return (closest+1,angular_dist(cat[closest][1],cat[closest][2],ra,dec))

#5

def angular_dist(RA1, dec1, RA2, dec2):
    # Convert to radians
    r1 = np.radians(RA1)
    d1 = np.radians(dec1)
    r2 = np.radians(RA2)
    d2 = np.radians(dec2)
    
    deltar = np.abs(r1 - r2)
    deltad = np.abs(d1 - d2)
    angle = 2*np.arcsin(np.sqrt(np.sin(deltad/2)**2 
                        + np.cos(d1)*np.cos(d2)*np.sin(deltar/2)**2))
    
    # Convert back to degrees
    return np.degrees(angle)

def crossmatch(cat1, cat2, max_radius):
    matches = []
    no_matches = []
    for id1, ra1, dec1 in cat1:
        closest_dist = np.inf
        closest_id2 = None
        for id2, ra2, dec2 in cat2:
            dist = angular_dist(ra1, dec1, ra2, dec2)
            if dist < closest_dist:
                closest_id2 = id2
                closest_dist = dist
        
        # Ignore match if it's outside the maximum radius
        if closest_dist > max_radius:
            no_matches.append(id1)
        else:
            matches.append((id1, closest_id2, closest_dist))

    return matches, no_matches



#%%
#Cross-matching with k-d trees

#1
import numpy as np
import time

def angular_dist(r1,d1,r2,d2):
#  r1,d1,r2,d2 = np.radians(ra1),np.radians(dec1),np.radians(ra2),np.radians(dec2)
  b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
  a = np.sin(np.abs(d1 - d2)/2)**2
  d = 2*np.arcsin(np.sqrt(a + b))
  return np.degrees(d)
def convertor(c):
  c = np.radians(c)
  return c
      
def crossmatch(cat1,cat2,min_allowed_d):
  cat1,cat2 = convertor(cat1),convertor(cat2)
  matches = []
  non_matches = []
  start_time = time.perf_counter()
  for i in range(len(cat1)):
    minId = 0
    d_min = np.Inf
    for j in range(len(cat2)):
      dist = angular_dist(cat1[i][0],cat1[i][1],cat2[j][0],cat2[j][1])
      if dist < d_min:
        minId = j
        d_min = dist
    if d_min < min_allowed_d:
      matches.append((i,minId,d_min))
    else:
      non_matches.append(i)
  run_time = time.perf_counter() - start_time
  return matches,non_matches,run_time


#2

def angular_dist(r1,d1,r2,d2):
#  r1,d1,r2,d2 = np.radians(ra1),np.radians(dec1),np.radians(ra2),np.radians(dec2)
  b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
  a = np.sin(np.abs(d1 - d2)/2)**2
  d = 2*np.arcsin(np.sqrt(a + b))
  return d

def convertor(c):
  c = np.radians(c)
  return c

def crossmatch(cat1,cat2,min_allowed_d):
  cat1,cat2,min_allowed_d = convertor(cat1),convertor(cat2),np.radians(min_allowed_d)
  matches = []
  non_matches = []
  start_time = time.perf_counter()
  for i in range(len(cat1)):
    ra1s,dec1s = np.full(shape = (len(cat2)), fill_value = cat1[i][0]),np.full(shape = (len(cat2)), fill_value = cat1[i][1])
    ra2s,dec2s = cat2[:,0],cat2[:,1]
    dists = angular_dist(ra1s,dec1s,ra2s,dec2s)
    min_dist = np.min(dists)
    if min_dist < min_allowed_d:
      closest_id2 = np.argmin(dists)
      matches.append((i,closest_id2,np.degrees(min_dist)))
    else:
      non_matches.append(i)
  run_time = time.perf_counter() - start_time
  return matches,non_matches,run_time


#3
import numpy as np
import time

def angular_dist(r1,d1,r2,d2):
#  r1,d1,r2,d2 = np.radians(ra1),np.radians(dec1),np.radians(ra2),np.radians(dec2)
  b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
  a = np.sin(np.abs(d1 - d2)/2)**2
  d = 2*np.arcsin(np.sqrt(a + b))
  return d


def crossmatch(cat1,cat2,min_allowed_d):
  cat1,cat2,min_allowed_d = np.radians(cat1),np.radians(cat2),np.radians(min_allowed_d)
  cat3 = np.zeros(shape = (len(cat2),3)) 
  for i in range(len(cat3)):
    cat3[i,0],cat3[i,1],cat3[i,2] = i,cat2[i,0],cat2[i,1]
  cat3 = cat3[cat3[:,2].argsort()]
  matches = []
  non_matches = []
  start_time = time.perf_counter()
  for i in range(len(cat1)):
    min_dist = np.Inf
    minId = cat3[0,0]
    r1,d1 = cat1[i,0],cat1[i,1]
    for j in range(len(cat3)):
      if cat3[j,2] <= d1 + min_allowed_d:
        r2,d2 = cat3[j,1],cat3[j,2]
        dist = angular_dist(r1,d1,r2,d2)
        if min_dist > dist:
          minId,min_dist = cat3[j,0],dist
      else:
        break
    if min_dist < min_allowed_d:
      matches.append((i,minId,np.degrees(min_dist)))
    else:
      non_matches.append(i)
  run_time = time.perf_counter() - start_time
  return matches,non_matches,run_time

#4

import numpy as np
import time

def angular_dist(r1,d1,r2,d2):
#  r1,d1,r2,d2 = np.radians(ra1),np.radians(dec1),np.radians(ra2),np.radians(dec2)
  b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2
  a = np.sin(np.abs(d1 - d2)/2)**2
  d = 2*np.arcsin(np.sqrt(a + b))
  return d


def crossmatch(cat1,cat2,min_allowed_d):
  cat1,cat2,min_allowed_d = np.radians(cat1),np.radians(cat2),np.radians(min_allowed_d)
  cat3 = np.zeros(shape = (len(cat2),3)) 
  for i in range(len(cat3)):
    cat3[i,0],cat3[i,1],cat3[i,2] = i,cat2[i,0],cat2[i,1]
  cat3 = cat3[cat3[:,2].argsort()]
  matches = []
  non_matches = []
  start_time = time.perf_counter()
  for i in range(len(cat1)):
    min_dist = np.Inf
    minId = cat3[0,0]
    r1,d1 = cat1[i,0],cat1[i,1]
    start_index = np.searchsorted(cat3[:,1],d1-min_allowed_d)
    for j in range(start_index,len(cat3)):
      if cat3[j,2] <= d1 + min_allowed_d:
        r2,d2 = cat3[j,1],cat3[j,2]
        dist = angular_dist(r1,d1,r2,d2)
        if min_dist > dist:
          minId,min_dist = cat3[j,0],dist
      else:
        break
    if min_dist < min_allowed_d:
      matches.append((i,minId,np.degrees(min_dist)))
    else:
      non_matches.append(i)
  run_time = time.perf_counter() - start_time
  return matches,non_matches,run_time

#5

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import time



def crossmatch(cat1,cat2,min_allowed_d):
  matches = []
  non_matches = []
  start_time = time.perf_counter()
  sky_cat1 = SkyCoord(cat1*u.degree, frame='icrs')
  sky_cat2 = SkyCoord(cat2*u.degree, frame='icrs')
  closest_ids, closest_dists, closest_dists3d = sky_cat1.match_to_catalog_sky(sky_cat2)
  closest_dists = closest_dists.value
  for i in range(len(cat1)):
    index_min = closest_ids[i]
    d_min = closest_dists[i]
    if d_min < min_allowed_d:
      matches.append((i,index_min,d_min))
    else:
      non_matches.append(i)
  run_time = time.perf_counter() - start_time
  return matches,non_matches,run_time




#%%

