
#Ashwin Goyal

#%%
#this files contains sql code, should not be run in python

#setting up your own database

#1

insert into Star values
(7115384,	3789,	27.384),
(8106973,	5810,	0.811),
(9391817,	6200,	0.958);

#2

update Planet set kepler_name = null
where not status = 'CONFIRMED';

delete from Planet where radius < 0;

#3


create table Planet(
  kepler_id INTEGER not null,
  koi_name varchar(15) unique not null,
  kepler_name varchar(15),
  status varchar(20) not null,
  radius float not null
);

insert into Planet values
(6862328,	'K00865.01', null,		'CANDIDATE',	119.021),
(10187017,	'K00082.05',	'Kepler-102 b',	'CONFIRMED',	5.286),
(10187017,	'K00082.04',	'Kepler-102 c',	'CONFIRMED',	7.071);

#4

create table Star(
  kepler_id integer primary key,
  t_eff integer not null,
  radius float not null
);

create table Planet(
  kepler_id integer references Star(kepler_id),
  koi_name varchar(20) primary key,
  kepler_name varchar(20),
  status varchar(20) not null,
  period float,
  radius float,
  t_eq integer
);

copy Star
from 'stars.csv' CSV;

copy Planet
from 'planets.csv' CSV;


#5

alter table Star
add column ra float,
add column decl float;


delete from Star;

copy Star 
from 'stars_full.csv' CSV;

select * from Star;


#%%

#Combining sql and python

#1

import psycopg2


def select_all(s):
  conn = psycopg2.connect(dbname='db', user='grok')
  cursor = conn.cursor()
# Execute an SQL query and receive the output
  cursor.execute('SELECT * from {};'.format(s))
  return cursor.fetchall()




#2



import numpy as np


def column_stats(t_name,c_name):
  conn = psycopg2.connect(dbname='db', user='grok')
  cursor = conn.cursor()
# Execute an SQL query and receive the output
  cursor.execute('SELECT {col} from {tab};'.format(col = c_name, tab = t_name))
  values =  cursor.fetchall()
  val_arr = np.array(values)
  return np.mean(val_arr), np.median(val_arr)


#3

# Write your query function here
import numpy as np

def query(f_name):
  arr = np.loadtxt(f_name,delimiter = ',', usecols = (0,2))
  index = np.where(arr[:,1] > 1)
  arr2 = arr[index]
  return arr2



#4

import numpy as np

def query(f_name):
  arr = np.loadtxt(f_name,delimiter = ',', usecols = (0,2))
  index = np.where(arr[:,1] > 1)
  arr2 = arr[index]
  sorts = np.argsort(arr2[:,1])
  arr3 = arr2[sorts]
  return arr3



#5
import numpy as np

def query(f_name1,f_name2):
  arr_stars = np.loadtxt(f_name1,delimiter = ',', usecols = (0,2))
  arr_planets = np.loadtxt(f_name2,delimiter = ',', usecols = (0,5))
  l = []
  for planet in arr_planets:
    for star in arr_stars:
      if planet[0] == star[0] and star[1] > 1:
        l.append([planet[1]/star[1]])
  arr = np.array(l)
  arr2 = arr[np.argsort(arr[:,0])]
  return arr2





#%%

