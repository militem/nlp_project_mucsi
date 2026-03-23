import requests
import json
import time
import os
import argparse

API_KEY = "4ff965cae7msh380e22645b4d329p1c31acjsn4583fc2e1378"
HOST = "booking-com15.p.rapidapi.com"
HEADERS = {"x-rapidapi-host": HOST, "x-rapidapi-key": API_KEY}
ARRIVAL_DATE = "2026-03-11"
DEPARTURE_DATE = "2026-03-12"
TIMEOUT = 15

def get_dest_id(city_name):
    url = f"https://{HOST}/api/v1/hotels/searchDestination"
    try:
        response = requests.get(url, headers=HEADERS, params={"query": city_name}, timeout=TIMEOUT)
        if response.status_code != 200: return None
        data = response.json().get('data', [])
        for item in data:
            if item.get('search_type') == 'city': return item.get('dest_id')
        return data[0].get('dest_id') if data else None
    except:
        return None

def get_hotel_details(hotel_id):
    url = f"https://{HOST}/api/v1/hotels/getHotelDetails"
    params = {"hotel_id": hotel_id, "arrival_date": ARRIVAL_DATE, "departure_date": DEPARTURE_DATE}
    try:
        res = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
        if res.status_code != 200: return "N/A", [], "N/A"
        d = res.json().get('data', {})
        address = d.get('address', 'N/A')
        f_list = d.get('facilities_block', {}).get('facilities', [])
        services = [f.get('name') for f in f_list][:10]
        url_book = d.get('url', f"https://www.booking.com/hotel/es/{hotel_id}.html")
        return address, services, url_book
    except:
        return "N/A", [], "N/A"

def load_existing_names(file_path):
    if not os.path.exists(file_path): return set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return {item.get("name") for item in json.load(f) if "name" in item}
    except:
        return set()

def save_data(new_hotels, file_path):
    if not new_hotels: return
    data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except: pass
    data.extend(new_hotels)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ciudad", choices=['M', 'B', 'P'], required=True)
    parser.add_argument("-n", "--cantidad", type=int, required=True)
    args = parser.parse_args()
    
    city_map = {'M': 'Madrid', 'B': 'Barcelona', 'P': 'Bilbao'}
    city_name = city_map[args.ciudad]
    json_file = "booking_results.json"
    processed_names = load_existing_names(json_file)
    
    dest_id = get_dest_id(city_name)
    if not dest_id: return

    all_hotels = []
    page = 1
    
    while len(all_hotels) < args.cantidad:
        params = {
            "dest_id": dest_id, "search_type": "CITY", "arrival_date": ARRIVAL_DATE, 
            "departure_date": DEPARTURE_DATE, "adults": "2", "page_number": str(page)
        }
        try:
            res = requests.get(f"https://{HOST}/api/v1/hotels/searchHotels", headers=HEADERS, params=params, timeout=TIMEOUT)
            hotels = res.json().get('data', {}).get('hotels', [])
            if not hotels: break
            
            for h in hotels:
                if len(all_hotels) >= args.cantidad: break
                prop = h.get('property', {})
                name = prop.get('name')
                
                if not name or name in processed_names: continue
                
                addr, serv, url = get_hotel_details(h.get('hotel_id'))
                price = prop.get('priceBreakdown', {}).get('grossPrice', {}).get('value', 'N/A')
                
                all_hotels.append({
                    "hotel_id": h.get('hotel_id'),
                    "city": city_name,
                    "name": name,
                    "price": round(price, 2) if isinstance(price, (int, float)) else price,
                    "rating": prop.get('reviewScore', 'N/A'),
                    "address": addr,
                    "services": serv,
                    "url": url
                })
                time.sleep(1)
            page += 1
        except:
            break

    save_data(all_hotels, json_file)

if __name__ == "__main__":
    main()