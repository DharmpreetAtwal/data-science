# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: % Dharm Atwal
"""

from os.path import join
from pyspark.sql import SparkSession
from pyspark.sql.functions import floor, sum, count, to_date, hour
import argparse


def sum_round2(col: str):
    return floor(sum(col) * 100) / 100

# %%
if __name__ == "__main__":
    # Getting the arguments AWS will pass to this script
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_source", type=str)
    parse.add_argument("--output_uri", type=str)
    args = parse.parse_args()
    
    data_source = args.data_source
    output_uri = args.output_uri
    
    spark = SparkSession.builder.appName("NYC YELLOW TAXI").enableHiveSupport().getOrCreate()
        
    # %%
    try:
        df = spark.read.parquet(data_source)
            
        # df = spark.read.parquet("C:/Users/coolb/Downloads/Coding/data-science/nyc-taxi/data/yellow_tripdata_2023-01.parquet")
        # df = df.sample(fraction=0.001, seed=42)
        
        df.createOrReplaceTempView("taxi_trip")
        
    # %%
        
        # Vendor1 Trip Count
        # df_vendor = spark.sql(
        # """
            
        #     SELECT COUNT(VendorId) as Vendor1Count 
        #     FROM taxi_trip
        #     WHERE VendorID == 1
        
        # """)
        
        df_vendor = df \
            .groupBy("VendorID") \
            .agg((count("*")).alias("TripCount"))
    
        df_vendor.show()
        df_vendor.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(join(output_uri, "df_vendor.csv"))
    
    # %%
    
        # Routes and their TripCount, TotalRevenue, TotalDistance, TotalPassengers
        # df_routes = spark.sql(
        # """
        
        #     SELECT PULocationID, DOLocationID, 
        #     COUNT(1) as TripCount,
        #     FLOOR(SUM(Total_amount) * 100) / 100 as TotalRevenue,
        #     SUM(Trip_distance) as TotalDistance,
        #     SUM(Passenger_count) as TotalPassengers
        #     FROM taxi_trip
        #     GROUP BY PULocationID, DOLocationID
        #     ORDER BY TotalRevenue DESC
        
        # """)
    
        df_routes = df \
            .groupBy("PULocationID", "DOLocationID") \
            .agg(
                    (count("*")).alias("TripCount"),
                    (sum_round2("Total_amount")).alias("TotalRevenue"),
                    (sum("Trip_distance").alias("TotalDistance")),
                    (sum("Passenger_count")).alias("TotalPassengers")) \
            .orderBy("TotalRevenue", ascending=False)
    
        df_routes.show(vertical=True)
        df_routes.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(join(output_uri, "df_routes.csv"))
        
    # %%
        
        # General Revenue source breakdown
        # df_rev = spark.sql(
        # """
        
        #     SELECT 
        #     FLOOR(SUM(Fare_amount) * 100) / 100 as TotalFare,
        #     FLOOR(SUM(Extra) * 100) / 100 as TotalExtra,        
        #     FLOOR(SUM(MTA_tax) * 100) / 100 as TotalMTA,
        #     FLOOR(SUM(Improvement_surcharge) * 100) / 100 as TotalImprovementSurcharge,
        #     FLOOR(SUM(Tip_amount) * 100) / 100 as TotalTipAmount,
        #     FLOOR(SUM(Tolls_amount) * 100) / 100 as TotalTollsAmount,
        #     FLOOR(SUM(Congestion_Surcharge) * 100) / 100 as TotalCongestionSurcharge,   
        #     FLOOR(SUM(Airport_fee) * 100) / 100 as TotalAirportFee
        #     FROM taxi_trip
            
        # """)
        
        df_rev = df.select(
            (sum_round2("Fare_amount")).alias("TotalFare"),
            (sum_round2("Extra")).alias("TotalExtra"),
            (sum_round2("MTA_tax")).alias("TotalMTA"),
            (sum_round2("Improvement_surcharge")).alias("TotalImprovementSurcharge"),
            (sum_round2("Tip_amount")).alias("TotalTipAmount"),
            (sum_round2("Tolls_amount")).alias("TotalTollsAmount"),
            (sum_round2("Congestion_Surcharge")).alias("TotalCongestionSurcharge"),
            (sum_round2("Airport_fee")).alias("TotalAirportFee"))
        
        df_rev.show(vertical=True)
        df_rev.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(join(output_uri, "df_rev.csv"))
        
    # %%
        
        # Revenue breakdown for each RateCodeID
        # df_rate_code = spark.sql(
        # """
        
        #     SELECT RateCodeID, 
        #     FLOOR(SUM(Total_amount) * 100) / 100 as TotalRevenue
        #     FROM taxi_trip
        #     GROUP BY RateCodeID
        #     ORDER BY RateCodeID
        
        # """)
        
        df_rate_code = df \
            .groupBy("RateCodeID") \
            .agg((sum_round2("Total_amount")).alias("TotalRevenue")) \
            .orderBy("RateCodeID")
        
        df_rate_code.show()
        df_rate_code.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(join(output_uri, "df_rate_code.csv"))
        
    # %%
        
        # Revenue breakdown for each PaymentType
        # df_payment_type = spark.sql(
        # """
        
        #     SELECT payment_type,
        #     FLOOR(SUM(Total_amount) * 100) / 100 as TotalRevenue
        #     FROM taxi_trip
        #     GROUP BY payment_type
        #     ORDER BY payment_type
        
        # """)
        
        df_payment_type = df \
            .groupBy("payment_type") \
            .agg((sum_round2("Total_amount")).alias("TotalRevenue")) \
            .orderBy("payment_type")
            
        df_payment_type.show()
        df_payment_type.write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(join(output_uri, "df_payment_type.csv"))
        
    # %%
    
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
            .csv(join(output_uri, "df_day_hour.csv"))
            
    finally:
        spark.stop()
