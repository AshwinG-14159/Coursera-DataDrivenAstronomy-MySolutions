
# Ashwin goyal


#%%
#intro to coding on grok
def greet(person):
  return "Hello, "+ person + "!"
#%%

#calculating the mean stack

#1
def calculate_mean(l):
  return sum(l)/len(l)

#2


import numpy as np

def calc_stats(name):
  data = np.loadtxt(name,delimiter = ',')
  mean = np.mean(data)
  median = np.median(data)
  return round(mean,1),round(median,1)

#3

def mean_datasets(l):
  arr = np.loadtxt(l[0],delimiter = ",")
  arr -= arr
  for i in l:
    arr += np.loadtxt(i,delimiter = ",")
  arr = arr/len(l)
  for i in range(len(arr)):
    for j in range(len(arr[i])):
      arr[i,j] = round(arr[i,j],1)
  return arr

from astropy.io import fits

#4

def load_fits(t):
  hdulist = fits.open(t)
  data = hdulist[0].data
  res = np.where(data == np.amax(data))
  val = list(zip(res[0],res[1]))[0]
  return val

#5

import matplotlib.pyplot as plt

def mean_fits(l):
  arr = fits.open(l[0])[0].data
  arr -= arr
  for i in l:
    hdulist = fits.open(i)
    data = hdulist[0].data
    arr += data
  arr = arr/len(l)
  return arr



#%%

#calculating the median stack

#1
def list_stats(l):
  fluxes = list(l)
  fluxes.sort()
  if len(l)%2 == 0:
    mid = len(fluxes)//2
    median = (fluxes[mid - 1] + fluxes[mid])/2
  else:
    mid = len(fluxes)//2
    median = fluxes[mid]
  s = 0
  for i in l:
    s += i
  mean = s/len(l)
  return median,mean

#2

import numpy as np
import statistics
import time

def time_stat(func, size, ntrials):
  s = 0
  
  for i in range(ntrials):
  # the time to generate the random array should not be included
    data = np.random.rand(size)
    start = time.perf_counter()
  # modify this function to time func with ntrials times using a new random array each time
    res = func(data)
    sam = time.perf_counter() - start
    s += sam
  # return the average run time
  
  return s/ntrials

#3



def median_fits(l):
  t_start = time.perf_counter()
  list_of_img = []
  for i in l:
    hdulist = fits.open(i)
    data = hdulist[0].data
    list_of_img.append(data)
  img_stack = np.stack(list_of_img)
  storage = img_stack.nbytes/1024
  med_arr = np.median(img_stack, axis = 0)
  t = time.perf_counter()-t_start
  return (med_arr,t,storage)


#4

def median_bins(values,B):
  mean = np.mean(values)
  sd = np.std(values)
  minval = mean-sd
  width = 2*sd/B
  bin_arr = np.zeros(shape = (B))
  low_outliers = 0
  for i in values:
    if i < minval:
      low_outliers+=1
    for j in range(B):
      if j*width <= i-minval < (j+1)*width:
        bin_arr[j] += 1
  return mean,sd,low_outliers,bin_arr

def median_approx(values,B):
  mean,sd,low_outliers,bin_arr = median_bins(values,B)
  val_to_reach = ((len(values)+1)/2)-low_outliers
  sum = 0
  for i in range(len(bin_arr)):
    sum += bin_arr[i]
    if sum >= val_to_reach:
      return mean - sd + 2*sd*(i+1/2)/B
  
  return mean-sd + 2*sd*(len(bin_arr) - 1/2)/B

#5

import time, numpy as np
from astropy.io import fits
from helper import running_stats


def median_bins_fits(filenames, B):
  # Calculate the mean and standard dev
  mean, std = running_stats(filenames)
    
  dim = mean.shape # Dimension of the FITS file arrays
    
  # Initialise bins
  left_bin = np.zeros(dim)
  bins = np.zeros((dim[0], dim[1], B))
  bin_width = 2 * std / B 

  # Loop over all FITS files
  for filename in filenames:
      hdulist = fits.open(filename)
      data = hdulist[0].data

      # Loop over every point in the 2D array
      for i in range(dim[0]):
        for j in range(dim[1]):
          value = data[i, j]
          mean_ = mean[i, j]
          std_ = std[i, j]

          if value < mean_ - std_:
            left_bin[i, j] += 1
                
          elif value >= mean_ - std_ and value < mean_ + std_:
            bin = int((value - (mean_ - std_))/bin_width[i, j])
            bins[i, j, bin] += 1

  return mean, std, left_bin, bins


def median_approx_fits(filenames, B):
  mean, std, left_bin, bins = median_bins_fits(filenames, B)
    
  dim = mean.shape # Dimension of the FITS file arrays
    
  # Position of the middle element over all files
  N = len(filenames)
  mid = (N + 1)/2
	
  bin_width = 2*std / B
  # Calculate the approximated median for each array element
  median = np.zeros(dim)   
  for i in range(dim[0]):
    for j in range(dim[1]):    
      count = left_bin[i, j]
      for b, bincount in enumerate(bins[i, j]):
        count += bincount
        if count >= mid:
          # Stop when the cumulative count exceeds the midpoint
          break
      median[i, j] = mean[i, j] - std[i, j] + bin_width[i, j]*(b + 0.5)
      
  return median


#%%







