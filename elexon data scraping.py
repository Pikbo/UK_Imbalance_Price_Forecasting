# ADG EFFICIENCY
# 2016-01-17
# this script pulls data off Elexon

import os
import pandas as pd
import numpy as np
import sqlite3

import urllib.request 
from lxml import objectify
from collections import OrderedDict

from pytz import timezone 

# setting base & SQL directory
base_path = os.path.dirname(os.path.abspath(__file__))
# setting up SQL connection
sql_name = os.path.join(base_path,'Data','ELEXON DATA' + "." + 'sqlite')
conn = sqlite3.connect(sql_name)
c = conn.cursor()

def BMRS_GetXML(**kwargs):
    # this function returns an XML object with our data for the given settlementdate & report keyword args
    url = 'https://api.bmreports.com/BMRS/{report}/v1?APIKey=***YOUR API KEY HERE***&ServiceType=xml'.format(**kwargs)
    for key, value in kwargs.items():
        if key not in ['report']: # report is removed because it is not a keyword arg, appears earlier in the URL
            a = "&%s=%s" % (key, value)
            url = url + a
    print(url)
    xml = objectify.parse(urllib.request.urlopen(url,timeout=500))
    return xml

def BMRS_Dataframe(**kwargs):
    # this function processes the XML object into a dataframe
    # navigates through to find the item in responseList
    # finds the root tag and builds an ordered dictionary, used to find the children tags     
    result = None
    numerror = 0
    while result is None:
        try:
            tags = []
            output = []
            for root in BMRS_GetXML(**kwargs).findall("./responseBody/responseList/item/"):
                tags.append(root.tag)
            tag = OrderedDict((x, 1) for x in tags).keys()
            df = pd.DataFrame(columns=tag)
            for root in BMRS_GetXML(**kwargs).findall("./responseBody/responseList/item"):
                data = root.getchildren()
                output.append(data)
            df = pd.DataFrame(output, columns=tag)                
            return df
        except Exception as exception:
            print(type(exception).__name__ )
            numerror = numerror + 1
            print(numerror)
            assert type(exception).__name__ == 'NameError'
                     
days = 366 # number of days to grab data for
startdate = '2016-01-01'
span = pd.date_range(startdate, periods = days, freq = 'D')
# variables used to track clocks channging & missing half hour periods
num_clockchange = 0
num_missing_HH = 0

# blank data frame to hold output of iteration through reports 
output = pd.DataFrame()

# setting error count metrics to zero
num_null = 0

# creating list of time zone changes
uk_tz = timezone("Europe/London")
transition_times = uk_tz._utc_transition_times[1:] # dropping first one as its garbage
transition_times = pd.to_datetime(transition_times) # converting to date time stamps

# for loop to iterate through different report numbers
for r in range(0,2):

    # blank data frame to hold output of iteration through days
    output_days = pd.DataFrame()
    # for loop to iterate through days
    for x in range(0,days):
        data_DF = pd.DataFrame()
        # setting date 
        SettlementDate = ''
        SettlementDate = str(span[x])
        SettlementDate = SettlementDate.rstrip()
        SettlementDate = SettlementDate.rstrip('00:00:00')
        SettlementDate = SettlementDate.rstrip()
        print(SettlementDate)
        SettlementDate_DT = pd.to_datetime(SettlementDate)

        # dictionary of reports to iterate through
        imbaprice = {'report' : 'B1770', 'SettlementDate':SettlementDate, 'Period' : '*','Data col name':'imbalancePriceAmountGBP'}
        imbavol = {'report':'B1780', 'SettlementDate':SettlementDate, 'Period':'*','Data col name':'imbalanceQuantityMAW'}
        reports = [imbaprice,imbavol]

        # setting our report
        report_info_D = reports[r]

        input_D = dict(report_info_D)
        del input_D['Data col name']
        
        # collecting data using function that grabs XML
        data = BMRS_Dataframe(**input_D)

        # converting data to DataFrame
        data_DF = pd.DataFrame(data)
        
        # exception for the price report (has two rows for every SP - one SBP one SSP)
        if report_info_D["report"] == 'B1770':
            data_DF = data_DF.iloc[::2] # removing every second row
        
        # sorting values
        data_DF.sort_values(by=['settlementDate', 'settlementPeriod'], ascending=[True, True], inplace=True)
        
        print('Original data is of length ' + str(len(data)))
        
        # removing duplicate settlement periods
        data_DF = data_DF.drop_duplicates(subset = 'settlementPeriod')
        
        # looking for missing settlement periods
        # creating a list of our dates where daylight saving changes
        dst_dates = []
        for time in transition_times:
            if time.year == SettlementDate_DT.year:
                time = pd.to_datetime(time.date())
                dst_dates.append(time)  
        print(dst_dates)
        
        # setting default index length
        index_length = 48
        
        # taking advantage of fact that first date = 46 HH, second date = 50
        if SettlementDate_DT == dst_dates[0]:
            index_length = 46
            
        if SettlementDate_DT == dst_dates[1]:
            index_length = 50
        
        print(index_length)
        
        # creating list of our settlementperiods
        SP = list(range(1,index_length+1))
        
        # creating our index of time stamps
        if (index_length==46):
            num_clockchange = num_clockchange + 1
            print('clocks changed')
            index = pd.date_range(SettlementDate, periods = index_length, freq = '30min') 
            index = index.tz_localize('UTC')    
            
        elif index_length == 48:
            print('normal day')
            index = pd.date_range(SettlementDate, periods = index_length, freq = '30min') 
            index = index.tz_localize('Europe/London')
            index = index.tz_convert('UTC')

        elif index_length == 50:
            num_clockchange = num_clockchange + 1
            print('clocks changed')
            delta = pd.to_timedelta('-1 hour')
            start_date = pd.Timestamp(SettlementDate)
            start_date = start_date + delta
            index = pd.date_range(start_date, periods = index_length, freq = '30min') 
            index = index.tz_localize('UTC')
        
        # setting report data name
        report_data_name = str(report_info_D['Data col name'])
        
        # creating a dataframe of settlement periods 
        SP_DF = pd.DataFrame(data = SP,columns=['settlementPeriod'])
        
        # saving raw data before any cleaning
        raw_DF = data_DF
        
        # merging with our settlement period data frame
        # this allows any missing settlement periods to be identified
        data_DF = data_DF.merge(SP_DF,'outer')
        data_DF.sort_values(by=['settlementPeriod'], ascending=[True], inplace=True)
        
        # counting any null values
        num_null = data_DF[report_data_name].isnull().sum() + num_null
        
        # creating copies of the index for saving the time stamps in different time zones
        TS_UK = index.copy()
        TS_UTC = index.copy()
        
        # removing time zone info from main index object for saving in SQL
        index = index.tz_localize(None)
        data_DF = data_DF.set_index(index)
        
        # creating series objects of the different time zones
        TS_UK = pd.Series(TS_UK.tz_convert('Europe/London'),index=index,name='UK TIME')
        TS_UTC = pd.Series(TS_UTC.tz_convert('UTC'),index=index,name='UTC TIME')
        
        # filling in missing values
        data_DF[report_data_name] = data_DF[report_data_name].fillna(data_DF[report_data_name].mean())
        
        # adding in UK and UTC time zones into dataframe
        data_DF = pd.concat([TS_UK,TS_UTC,data_DF],axis=1)        
        
        # saving the report into SQL
        report_name = report_info_D['report']
        data_DF = data_DF.astype(str)
        data_DF.to_sql(name=report_name,con=conn,flavor='sqlite',if_exists ='append')
       
        # appending the results of this day to the output for all days dataframe
        output_days = output_days.append(data_DF)
   
    # creating a master dataframe for saving to CSV
    output = pd.concat([output,output_days], axis=1)

# saving all data to CSV
csv_name = os.path.join(base_path,'Data','output' + "." + 'csv')
output.to_csv(csv_name)    

print('Number of missing values was ' + str(num_null))   
        
            
        
    
        
        
        
        
