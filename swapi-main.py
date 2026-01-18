import requests

response = requests.get("https://swapi.info/api/")
if response.status_code == 200:
    categories = response.json()
    print("\nAvailable Star Wars Categories:")
    for i, category in enumerate(categories.keys(), 1):
      print(f"{i}. {category.capitalize()}")
    print("-" * 30 + "\n")
else: 
  print(f"Failed to fetch categories. Status code: {response.status_code}")

raw_option = input("What star wars data would you like to explore? ").strip().lower()

options = {
    1: "films",
    2: "people",
    3: "planets",
    4: "species",
    5: "vehicles",
    6: "starships",
}

valid_options = set(options.values())
if raw_option.isdigit():
    option = options.get(int(raw_option))
else:
    option = raw_option if raw_option in valid_options else None



def fetch_data(option):
  data = []

  url = (f"https://swapi.info/api/{option}/")
  try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    print(len(data))
  except requests.HTTPError as e:
    return None
    print(f"Error: {e}")

  return data

data = fetch_data(option)

if data:
  for x in data:
    print(x.get("name") or x.get("title"))  
else:
  print("Unable to download data")


fetch_data(option)


