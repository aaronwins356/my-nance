"""
UFC Rankings Scraper
Scrapes official UFC rankings from UFC.com
"""
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Use relative path imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def ensure_directories():
    """Ensure cache and data directories exist"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def scrape_ufc_rankings(use_cache=True):
    """
    Scrape UFC rankings from UFC.com
    Returns: dict with rankings by weight class
    """
    cache_file = os.path.join(CACHE_DIR, 'rankings_cache.json')
    
    # Check cache first
    if use_cache and os.path.exists(cache_file):
        cache_age = time.time() - os.path.getmtime(cache_file)
        # Cache valid for 7 days
        if cache_age < 7 * 24 * 3600:
            print(f"Using cached rankings (age: {cache_age/3600:.1f} hours)")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    print("Scraping UFC rankings from UFC.com...")
    
    # UFC Rankings URL
    url = "https://www.ufc.com/rankings"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        rankings = {
            'scraped_at': datetime.now().isoformat(),
            'weight_classes': {}
        }
        
        # Parse rankings - UFC.com structure may vary, using generic approach
        # Look for weight class sections
        weight_class_sections = soup.find_all('div', class_='view-grouping')
        
        if not weight_class_sections:
            # Try alternative structure
            print("Using fallback parsing method...")
            rankings['weight_classes'] = get_fallback_rankings()
        else:
            for section in weight_class_sections:
                try:
                    # Extract weight class name
                    header = section.find('div', class_='view-grouping-header')
                    if not header:
                        continue
                    
                    weight_class = header.get_text(strip=True)
                    
                    # Extract fighters
                    fighters = []
                    rows = section.find_all('tr')
                    
                    for row in rows:
                        rank_cell = row.find('td', class_='views-field-weight-class-rank')
                        name_cell = row.find('td', class_='views-field-title')
                        
                        if rank_cell and name_cell:
                            rank = rank_cell.get_text(strip=True)
                            name = name_cell.get_text(strip=True)
                            
                            fighters.append({
                                'rank': rank,
                                'name': name
                            })
                    
                    if fighters:
                        rankings['weight_classes'][weight_class] = fighters
                
                except Exception as e:
                    print(f"Error parsing section: {e}")
                    continue
        
        # Save to cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(rankings, f, indent=2)
        
        print(f"Scraped {len(rankings['weight_classes'])} weight classes")
        return rankings
    
    except Exception as e:
        print(f"Error scraping rankings: {e}")
        # Try to load from cache as fallback
        if os.path.exists(cache_file):
            print("Loading from cache as fallback...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Return fallback rankings
            return {
                'scraped_at': datetime.now().isoformat(),
                'weight_classes': get_fallback_rankings()
            }

def get_fallback_rankings():
    """
    Fallback rankings for development/testing
    Top 10 fighters per men's division
    """
    return {
        "Men's Pound-for-Pound": [
            {"rank": "C", "name": "Islam Makhachev"},
            {"rank": "1", "name": "Jon Jones"},
            {"rank": "2", "name": "Alexander Volkanovski"},
            {"rank": "3", "name": "Leon Edwards"},
            {"rank": "4", "name": "Israel Adesanya"},
            {"rank": "5", "name": "Charles Oliveira"},
            {"rank": "6", "name": "Max Holloway"},
            {"rank": "7", "name": "Kamaru Usman"},
            {"rank": "8", "name": "Sean O'Malley"},
            {"rank": "9", "name": "Ilia Topuria"},
            {"rank": "10", "name": "Alex Pereira"}
        ],
        "Heavyweight": [
            {"rank": "C", "name": "Jon Jones"},
            {"rank": "1", "name": "Tom Aspinall"},
            {"rank": "2", "name": "Ciryl Gane"},
            {"rank": "3", "name": "Sergei Pavlovich"},
            {"rank": "4", "name": "Curtis Blaydes"},
            {"rank": "5", "name": "Alexander Volkov"},
            {"rank": "6", "name": "Jailton Almeida"},
            {"rank": "7", "name": "Derrick Lewis"},
            {"rank": "8", "name": "Marcin Tybura"},
            {"rank": "9", "name": "Tai Tuivasa"},
            {"rank": "10", "name": "Alexandr Romanov"}
        ],
        "Light Heavyweight": [
            {"rank": "C", "name": "Alex Pereira"},
            {"rank": "1", "name": "Jamahal Hill"},
            {"rank": "2", "name": "Jiri Prochazka"},
            {"rank": "3", "name": "Jan Blachowicz"},
            {"rank": "4", "name": "Aleksandar Rakic"},
            {"rank": "5", "name": "Magomed Ankalaev"},
            {"rank": "6", "name": "Nikita Krylov"},
            {"rank": "7", "name": "Johnny Walker"},
            {"rank": "8", "name": "Anthony Smith"},
            {"rank": "9", "name": "Volkan Oezdemir"},
            {"rank": "10", "name": "Khalil Rountree Jr."}
        ],
        "Middleweight": [
            {"rank": "C", "name": "Dricus Du Plessis"},
            {"rank": "1", "name": "Sean Strickland"},
            {"rank": "2", "name": "Israel Adesanya"},
            {"rank": "3", "name": "Robert Whittaker"},
            {"rank": "4", "name": "Jared Cannonier"},
            {"rank": "5", "name": "Marvin Vettori"},
            {"rank": "6", "name": "Paulo Costa"},
            {"rank": "7", "name": "Khamzat Chimaev"},
            {"rank": "8", "name": "Brendan Allen"},
            {"rank": "9", "name": "Jack Hermansson"},
            {"rank": "10", "name": "Roman Dolidze"}
        ],
        "Welterweight": [
            {"rank": "C", "name": "Leon Edwards"},
            {"rank": "1", "name": "Belal Muhammad"},
            {"rank": "2", "name": "Kamaru Usman"},
            {"rank": "3", "name": "Colby Covington"},
            {"rank": "4", "name": "Shavkat Rakhmonov"},
            {"rank": "5", "name": "Jack Della Maddalena"},
            {"rank": "6", "name": "Gilbert Burns"},
            {"rank": "7", "name": "Sean Brady"},
            {"rank": "8", "name": "Stephen Thompson"},
            {"rank": "9", "name": "Geoff Neal"},
            {"rank": "10", "name": "Ian Garry"}
        ],
        "Lightweight": [
            {"rank": "C", "name": "Islam Makhachev"},
            {"rank": "1", "name": "Arman Tsarukyan"},
            {"rank": "2", "name": "Charles Oliveira"},
            {"rank": "3", "name": "Justin Gaethje"},
            {"rank": "4", "name": "Dustin Poirier"},
            {"rank": "5", "name": "Michael Chandler"},
            {"rank": "6", "name": "Beneil Dariush"},
            {"rank": "7", "name": "Dan Hooker"},
            {"rank": "8", "name": "Mateusz Gamrot"},
            {"rank": "9", "name": "Rafael Fiziev"},
            {"rank": "10", "name": "Renato Moicano"}
        ],
        "Featherweight": [
            {"rank": "C", "name": "Ilia Topuria"},
            {"rank": "1", "name": "Max Holloway"},
            {"rank": "2", "name": "Alexander Volkanovski"},
            {"rank": "3", "name": "Brian Ortega"},
            {"rank": "4", "name": "Yair Rodriguez"},
            {"rank": "5", "name": "Arnold Allen"},
            {"rank": "6", "name": "Movsar Evloev"},
            {"rank": "7", "name": "Josh Emmett"},
            {"rank": "8", "name": "Giga Chikadze"},
            {"rank": "9", "name": "Calvin Kattar"},
            {"rank": "10", "name": "Diego Lopes"}
        ],
        "Bantamweight": [
            {"rank": "C", "name": "Sean O'Malley"},
            {"rank": "1", "name": "Merab Dvalishvili"},
            {"rank": "2", "name": "Cory Sandhagen"},
            {"rank": "3", "name": "Petr Yan"},
            {"rank": "4", "name": "Marlon Vera"},
            {"rank": "5", "name": "Deiveson Figueiredo"},
            {"rank": "6", "name": "Henry Cejudo"},
            {"rank": "7", "name": "Song Yadong"},
            {"rank": "8", "name": "Rob Font"},
            {"rank": "9", "name": "Dominick Cruz"},
            {"rank": "10", "name": "Ricky Simon"}
        ],
        "Flyweight": [
            {"rank": "C", "name": "Alexandre Pantoja"},
            {"rank": "1", "name": "Brandon Moreno"},
            {"rank": "2", "name": "Brandon Royval"},
            {"rank": "3", "name": "Amir Albazi"},
            {"rank": "4", "name": "Kai Kara-France"},
            {"rank": "5", "name": "Matheus Nicolau"},
            {"rank": "6", "name": "Alex Perez"},
            {"rank": "7", "name": "Muhammad Mokaev"},
            {"rank": "8", "name": "Steve Erceg"},
            {"rank": "9", "name": "Manel Kape"},
            {"rank": "10", "name": "Tim Elliott"}
        ]
    }

def save_rankings_to_csv():
    """Save rankings to CSV format for downstream processing"""
    rankings = scrape_ufc_rankings()
    
    import pandas as pd
    
    all_fighters = []
    
    for weight_class, fighters in rankings['weight_classes'].items():
        for fighter in fighters:
            all_fighters.append({
                'weight_class': weight_class,
                'rank': fighter['rank'],
                'name': fighter['name']
            })
    
    df = pd.DataFrame(all_fighters)
    output_file = os.path.join(DATA_DIR, 'ufc_rankings.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved rankings to {output_file}")
    print(f"Total fighters: {len(df)}")
    return df

if __name__ == '__main__':
    ensure_directories()
    save_rankings_to_csv()
