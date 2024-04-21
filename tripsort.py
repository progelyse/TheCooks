import pandas as pd

import numpy as np
import prestodb
import csv
import requests

csvpath="C:/Users/tommy/Downloads/"

api_key = 'AIzaSyDrUkuXqlmszCHrCnKbyuf97JgFH91pFtw' #google api key

conn = prestodb.dbapi.connect(
       host='na4.services.cloud.techzone.ibm.com',
       port=37980,
       user='ibmlhadmin',
       catalog='hive_data',
       schema='ontime',
       http_scheme='https',
       auth=prestodb.auth.BasicAuthentication("ibmlhadmin", "password")
)
conn._http_session.verify = csvpath+'lh-ssl-ts.crt'

cur = conn.cursor()
cur2 = conn.cursor()


cur.execute("SELECT * FROM hive_data.ontime.ontime limit 153 ")
rows = cur.fetchall()
rowsinfo = cur.description #extract header apo to selection
rowsinfo = [sublist[0] for sublist in rowsinfo] #filter the header name
rows.insert(0,rowsinfo) #add header back

cur2.execute("SELECT * FROM hive_data.taxi.taxirides limit 42327")
rows2= cur2.fetchall()
rowsinfo2 = cur2.description
rowsinfo2 = [sublist[0] for sublist in rowsinfo2]
rows2.insert(0,rowsinfo2)




#taxisort

#API KEY
#
durationlist=[]#keeps trip_id, duration and distance
count=0

#stelnei requests
breaknumber=10 # to stop after 100 requests cuz we have a big sample size to choose from

for i in rows2[1:]:
       latitude=i[2] #dropoff cords
       longitude=i[3]
       #print(latitude,longitude)
       origins = str(latitude) +" "+ str(longitude)
       if isinstance(latitude, float) and isinstance(longitude, float):
           # Define the origins and destinations
           destinations = 'chicago airport'

           # Construct the API URL
           url = f'https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins={origins}&destinations={destinations}&key={api_key}'

           # Make the API request
           response = requests.get(url)

           # Check if the request was successful
           if response.status_code == 200:
               data = response.json()
               # Extract and print the distance and duration
               distance = data['rows'][0]['elements'][0]['distance']['text']
               duration = data['rows'][0]['elements'][0]['duration']['text']
               print(f"Ride ID: {i[0]} Distance: {distance}, Duration: {duration},current longitude: {i[2]}, latitude={i[3]}")
               durationlist.insert(count,[i[0],duration,distance,i[2],i[3]])#inserts tripid,duration,distance,currentcords

           else:
               print(f"Failed to get distance matrix: {response.status_code}")

       count+=1
       if count==breaknumber:
           break;

sorted_a = sorted(durationlist, key=lambda x: (x[1], x[2]))

for item in sorted_a:
    print(item)



