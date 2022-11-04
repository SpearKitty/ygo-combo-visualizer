import requests, json

#thanks for being helpful ygoprodeck
API_URL_BASE = 'https://db.ygoprodeck.com/api/v7/cardinfo.php?'

API_ARC_URL  = 'https://db.ygoprodeck.com/api/v7/archetypes.php' #unused

API_VER_URL  = 'https://db.ygoprodeck.com/api/v7/checkDBVer.php'

#get db version
def get_db_version() -> str:
    req = requests.get(API_VER_URL)
    if req.status_code == 200:
        return json.loads(req.text[1:-1])['database_version']
    return f'Failed db version check. (response code: {req.status_code})'

#fuzzy search returning a list of dicts of only name, id, and (large) image url
def fuzzy_search(name:str) -> list[dict[str, str]] | None:
    req = requests.get(f'{API_URL_BASE}fname={name}')
    try:
        return [
            {
            'name' : data['name'], 
            'id' : data['card_images'][0]['id'], 
            'image' : data['card_images'][0]['image_url']
            }
                for data in json.loads(req.text)['data']
            ] if req.status_code == 200 else None
    except:
        return None

#fuzzy search returning full api response
def fuzzy_search_full(name:str) -> list[dict[str, str]] | None:
    req = requests.get(f'{API_URL_BASE}fname={name}')
    try:
        return json.loads(req.text)
    except:
        return None

#exact search returning exact card, None if there's no result returned
def exact_search(name:str) -> list[dict[str, str]] | None:
    req = requests.get(f'{API_URL_BASE}name={name}')
    try:
        return json.loads(req.text)
    except:
        return None