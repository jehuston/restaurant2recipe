import json
import requests
import pickle
import time
from bs4 import BeautifulSoup
from yelp.client import Client
from yelp.oauth1_authenticator import Oauth1Authenticator
from pymongo import MongoClient
from collections import defaultdict


def scrape_menu(business_id):
    '''
    INPUT:
    OUTPUT:

    '''
    link = 'https://www.yelp.com/menu/{0}'.format(business_id) ## need some error checking here

    return requests.get(link)

def get_menu_items(page):
    soup = BeautifulSoup(page.content)
    menu = defaultdict(list)
    for ms in soup.select('.menu-item-details'):
        item = ms.h4.text.strip() ## just menu items --> section titles are in .menu-section-header
        menu['items'].append(item)
        #print item
        if ms.p:
            desc= ms.p.text.strip()
            #print desc
            menu['descriptions'].append(desc)
        else:
            menu['descriptions'].append('no description available')
    return menu


def get_num_response_pages(client):
    params = {
        'term' : 'restaurants'#,
        # 'sort' : 2,
        # 'limit' : 20
        }
    bus_response = client.search('San Francisco', **params)
    num_pages = (bus_response.total/20) + 1 ## businesses returns 20 at a time so will need to repeat calls with offset
    return num_pages

def get_business_ids(client, page):
    ## For every page of responses: get business ids
    params = {
    'term' : 'restaurants',
    # 'sort' : 2,
    # 'limit' : 20,
    'offset' : page*20
    }
    response = client.search('San Francisco', **params)
    #print type(response.businesses)
    return response.businesses

def add_to_database(businesses, db):
    for i, bus in enumerate(businesses):
        if bus.menu_date_updated: ## check menu available on Yelp
            cursor = db.restaurants.find({'bus_id': bus.id}).limit(1) ## check if already in db
            if not cursor.count() > 0:
                bus_obj = {}
                ## for every business id: go to menu page and scrape dish names and descriptions
                response = scrape_menu(bus.id.encode('utf-8'))
                if response.status_code != 200:
                    continue ## go to next business

                ## parse menu
                menu = get_menu_items(response)

                ## build business obj dict
                bus_obj['bus_id'] = bus.id
                bus_obj['name'] = bus.name
                bus_obj['menu'] = menu
                #print bus_obj

                ##insert into db
                db.restaurants.insert_one(bus_obj)

                time.sleep(2)




if __name__ == '__main__':
    m_client = MongoClient()
    db = m_client['project']

    with open('credentials/yelp_config.json') as cred:
        creds = json.load(cred)
        auth = Oauth1Authenticator(**creds)
        client = Client(auth)

    pages = get_num_response_pages(client)
    #print pages
    for page in xrange(pages):
        responses = get_business_ids(client, page)
        add_to_database(responses, db)
        if page % 10 == 0:
            print "Just completed page ", page
