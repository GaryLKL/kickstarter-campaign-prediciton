import pandas as pd, numpy as np
import glob
from datetime import datetime, date
import json
import requests, random
from bs4 import BeautifulSoup
from time import sleep
import asyncio
import pyppeteer
from pyppeteer_fork import launch


data = []
with open("Analysis/Kickstarter_Tech.json", "r") as json_file:
    for line in json_file:
        data.append(json.loads(line))

table = pd.DataFrame([list(ele.values()) for ele in data], columns = list(data[0].keys()))

table["Description"] = [""] * len(table)
table["Image"] = [0] * len(table)
table["Video"] = [0] * len(table)


async def connect(table):
	# initialize a browser
	browser = await launch({'headless': False, "devtools": False, 'dumpio':False, 'autoClose':False,'args': ['--no-sandbox']})
	page = await browser.newPage()
	await page.setUserAgent('Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36')
	# count error times
	errortimes = 0
	#nrow_space = np.linspace(6000, 12000, 13, dtype = int)
	nrow_space = [12032]
	gap = 32
	for nrow in nrow_space:
	    project_description = []
	    project_scraped_name = "Analysis/project_scraped_" + str(nrow-gap) + "_" + str(nrow) + ".csv"
	    project_description_name = "Analysis/project_description_" + str(nrow-gap) + "_" + str(nrow) + ".txt"
	    for i in range(nrow-gap, nrow):
	        for test in range(2):
	            if test >= 2:
	                table["Description"][i] = ""
	                print("Error happened for three times.")
	                break
	            try:
	                url = table["url"][i]
	                await page.goto(url)
	                
	                try:
	                    await page.waitForSelector("div.rte__content", timeout = 2500)
	                    print("Success")
	                except:
	                    content = ""
	                    img = 0
	                    html_doc = await page.content()
	                    soup = BeautifulSoup(html_doc, "lxml")
	                    video = int(soup.find_all("video") != [])
	                    print("No Content. No Scraping needed")
	                    await asyncio.sleep(1)
	                    break
	                    
	                html_doc = await page.content()
	                soup = BeautifulSoup(html_doc, "lxml")
	                html_content = soup.find_all('div', attrs={"class":"rte__content"})[0].find_all("p")
	                content = ''.join([i.text for i in html_content])
	                img = int(soup.find_all('div', attrs={"class":"rte__content"})[0].find_all("figure") != [])
	                video = int(soup.find_all("video") != [])
	                # Sleep
	                await asyncio.sleep(random.randint(2,4))
	                break
	            except:
	                if errortimes > 25:
	                    await browser.close()
	                    await asyncio.sleep(1)
	                    browser = await launch({'headless': False,"devtools": False, 'dumpio':False, 'autoClose':False,'args': ['--no-sandbox']})
	                    page = await browser.newPage()
	                    await page.setUserAgent('Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36')
	                    errortimes = 0
	                    await asyncio.sleep(random.randint(1, 2))
	                else:
	                    page = await browser.newPage()
	                    await page.setUserAgent('Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36')
	                    await asyncio.sleep(random.randint(1,2))
	                errortimes += 1
	        project_description.append(str(i) + "|" + content)
	        table["Image"][i] = img
	        table["Video"][i] = video
	        
	    table.to_csv(project_scraped_name, index = False)
	    with open(project_description_name, "w") as file:
	        file.write("\n".join(project_description))


if __name__ == "__main__":
	result = asyncio.run(connect(table))
            
        