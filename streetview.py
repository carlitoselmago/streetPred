from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import os
from time import sleep

d = DesiredCapabilities.CHROME

d['goog:loggingPrefs'] = { 'browser':'ALL' }

def extractPanoData(driver):
    pdata={}
    pdata["id"]=driver.execute_script('return exporter[0].result.location.pano;')
    pdata["links"]=driver.execute_script('return exporter[0].result.links;')

    return pdata

driver = webdriver.Chrome(desired_capabilities=d)
driver.get('file:///'+os.getcwd()+"/selleniuminterface/index.html")
for entry in driver.get_log('browser'):
    print(entry)
sleep(4)
panoData=extractPanoData(driver)
print(panoData)
driver.close()
