# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: % Dharm Atwal
"""

from os.path import join
from pyspark.sql import SparkSession
from pyspark.sql.functions import floor, sum, count, col, to_date, hour
import argparse
import sys

def sum_round2(col: str):
    return floor(sum(col) * 100) / 100

# %%
# Getting the args AWS will pass to this script
parse = argparse.ArgumentParser()
parse.add_argument("--data_source", type=str)
parse.add_argument("--output_uri", type=str)
args = parse.parse_args()

data_source = args.data_source
output_uri = args.output_uri

spark = SparkSession.builder \
    .appName("NYC YELLOW TAXI") \
    .enableHiveSupport() \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# %%
try:
    lst = []
    for i in range(1, 13):
        try:
            uri = join(data_source, f'yellow_tripdata_2023-{i:02}.parquet')
            df_temp = spark.read.parquet(uri)
            df_temp = df_temp.withColumn("VendorID", col("VendorID").cast("bigint")) 
            
            lst.append(df_temp)
        except Exception as e:
            print(f"Error reading file for month {i}: {str(e)}")
            sys.exit(1)
    
    df = lst[0]
    for df_temp in lst[1:]:
        df = df.unionByName(df_temp)
    df.createOrReplaceTempView("taxi_trip")
        
    # Day-Hour Breakdown of TripCount, TotalRevenue, TotalPassengers, Total Distance
    df_day_hour = df \
        .groupBy((to_date("tpep_pickup_datetime")).alias("Date"), 
                 (hour("tpep_pickup_datetime")).alias("Hour")) \
        .agg(
                (count("*")).alias("TripCount"),
                (sum_round2("Total_amount")).alias("TotalRevenue"),
                (sum("Passenger_count")).alias("TotalPassengers"),
                (sum("Trip_distance")).alias("TotalDistance")) \
        .orderBy(to_date("tpep_pickup_datetime"), hour("tpep_pickup_datetime"))
    
    df_day_hour.show()
    df_day_hour.write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(join(output_uri, "df_day_hour"))
        
finally:
    spark.stop()
