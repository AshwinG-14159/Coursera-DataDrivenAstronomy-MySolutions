#Ashwin Goyal


#%%

#this file has code in SQL, should not be run in python


#writing your own sql queries

#1
SELECT radius,t_eff from STAR where radius > 1

#2

select kepler_id,t_eff from Star where t_eff between 5000 and 6000

#3

select kepler_name,radius from Planet 
where status = 'CONFIRMED' and radius between 1 and 3

#4
select min(radius),max(radius),avg(radius),stddev(radius) from Planet
where kepler_name is null

#5
select kepler_id,count(koi_name) 
from Planet 
group by kepler_id 
having count(koi_name) > 1 
order by count(koi_name) desc




#%%

#joining tables with sql

#1

select s.radius as "sun_radius",p.radius as "planet_radius"
from Star as s,Planet as p
where s.kepler_id = p.kepler_id and s.radius/p.radius > 1
order by s.radius desc

#2

select s.radius,count(p.koi_name)
from Planet as p join Star as s using(kepler_id)
where s.radius >1 and not s.radius = 1.046
group by s.radius
having count(p.koi_name) > 1
order by s.radius desc


#3

select s.kepler_id, s.t_eff, s.radius
from Star as s left outer join Planet as p using(kepler_id)
where p.koi_name is null
order by s.t_eff desc

#4
select round(avg(p.t_eq),1),min(s.t_eff),max(s.t_eff) 
from Planet as p join Star as s using(kepler_id)
where s.t_eff > (select avg(t_eff) 
  from Star)


#5

select p.koi_name,p.radius,s.radius
from Planet as p join Star as s using(kepler_id)
where s.kepler_id in (select kepler_id 
  from Star 
  order by radius desc 
  limit 5)





#%%






