# Enhanced script to collect NCAA basketball data for ALL teams
import pandas as pd
import requests
import json
import time
from tqdm import tqdm  # For progress bar

# 1. Fetch basic team data and save to CSV
def get_teams_to_csv():
    print("Fetching NCAA basketball teams data...")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams?limit=400"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        teams = []
        for item in data['sports'][0]['leagues'][0]['teams']:
            team = item['team']
            teams.append({
                'id': team['id'],
                'name': team['displayName'],
                'abbreviation': team.get('abbreviation', ''),
                'location': team.get('location', ''),
                'conference': team.get('conferenceId', '')
            })

        df = pd.DataFrame(teams)
        df.to_csv('ncaa_teams.csv', index=False)
        print(f"Saved {len(df)} teams to ncaa_teams.csv")
        return df
    except Exception as e:
        print(f"Error fetching teams: {e}")
        return pd.DataFrame()

# 2. Get statistics for ALL teams and save both raw and processed data
def get_all_team_stats(teams_df, batch_size=50):
    """
    Fetch statistics for all teams in batches to avoid rate limiting

    Args:
        teams_df: DataFrame containing team information
        batch_size: Number of teams to process before saving intermediate results
    """
    print(f"Fetching statistics for ALL teams ({len(teams_df)} total)...")
    if teams_df.empty:
        return

    all_stats = []
    raw_responses = []

    # Create a directory for individual team JSON files if needed
    import os
    if not os.path.exists('team_stats'):
        os.makedirs('team_stats')

    # Process all teams with a progress bar
    for idx, team in tqdm(teams_df.iterrows(), total=len(teams_df), desc="Fetching team stats"):
        team_id = team['id']
        team_name = team['name']

        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/statistics"
        try:
            response = requests.get(url)

            # Save response regardless of status to check what went wrong
            if response.status_code == 200:
                data = response.json()

                # Save individual team raw response
                with open(f"team_stats/team_{team_id}.json", 'w') as f:
                    json.dump(data, f, indent=2)

                # Add to combined raw responses
                raw_responses.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'raw_response': json.dumps(data)
                })

                # Extract stats for CSV
                team_stats = {'team_id': team_id, 'team_name': team_name}

                # Try to get record summary
                try:
                    if 'team' in data.get('results', {}):
                        team_stats['record'] = data['results']['team'].get('recordSummary', 'N/A')
                        team_stats['standing'] = data['results']['team'].get('standingSummary', 'N/A')
                except Exception as e:
                    pass  # Skip if record not available

                # Process statistics categories
                try:
                    for category in data.get('results', {}).get('stats', {}).get('categories', []):
                        category_name = category.get('name', '')
                        for stat in category.get('stats', []):
                            stat_name = stat.get('name', '')
                            stat_value = stat.get('value', '')
                            # Create name like "offensive_fieldGoalPct" or "defensive_steals"
                            column_name = f"{category_name}_{stat_name}"
                            team_stats[column_name] = stat_value
                except Exception as e:
                    print(f"  Error processing stats for {team_name}: {e}")

                all_stats.append(team_stats)
            else:
                print(f"  No stats available for {team_name} (status code: {response.status_code})")

            # Add a small delay to avoid rate limiting
            time.sleep(0.2)

            # Save intermediate results in batches
            if len(all_stats) % batch_size == 0:
                save_progress(all_stats, raw_responses, f"intermediate_{len(all_stats)}")

        except Exception as e:
            print(f"  Error fetching stats for {team_name}: {e}")

    # Final save
    save_progress(all_stats, raw_responses, "final")

    return all_stats

def save_progress(all_stats, raw_responses, suffix=""):
    """Save current progress to avoid losing data if script fails"""
    if all_stats:
        # Save processed stats
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(f'team_stats_{suffix}.csv', index=False)
        print(f"Saved statistics for {len(stats_df)} teams to team_stats_{suffix}.csv")

        # Save raw responses
        with open(f'team_stats_raw_{suffix}.json', 'w') as f:
            json.dump(raw_responses, f, indent=2)
            print(f"Saved raw JSON responses to team_stats_raw_{suffix}.json")

# 3. Get rankings data
def get_rankings_to_csv():
    print("Fetching NCAA basketball rankings...")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/rankings"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Save raw JSON
        with open('rankings_raw.json', 'w') as f:
            json.dump(data, f, indent=2)
            print("Saved raw rankings data to rankings_raw.json")

        # Extract all polls, not just AP
        rankings = []
        for poll in data.get('rankings', []):
            poll_name = poll.get('name', 'Unknown Poll')
            for rank in poll.get('ranks', []):
                rankings.append({
                    'poll': poll_name,
                    'rank': rank.get('current', ''),
                    'team_id': rank.get('team', {}).get('id', ''),
                    'team_name': rank.get('team', {}).get('name', ''),
                    'record': rank.get('recordSummary', ''),
                    'points': rank.get('points', ''),
                    'trend': rank.get('trend', '')
                })

        if rankings:
            df = pd.DataFrame(rankings)
            df.to_csv('ncaa_rankings.csv', index=False)
            print(f"Saved {len(df)} team rankings to ncaa_rankings.csv")
            return df
        else:
            print("No ranking data found")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching rankings: {e}")
        return pd.DataFrame()

# 4. Get bracketology data
def get_bracketology_to_json():
    print("Fetching NCAA bracketology data...")
    url = "https://site.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/bracketology"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            with open('bracketology_raw.json', 'w') as f:
                json.dump(data, f, indent=2)
                print("Saved raw bracketology data to bracketology_raw.json")

            # Process into CSV for easier analysis
            try:
                teams = []
                for region in data.get('regions', []):
                    region_name = region.get('name', '')
                    for seed in region.get('teams', []):
                        seed_num = seed.get('seedNum', '')
                        for team in seed.get('teams', []):
                            teams.append({
                                'region': region_name,
                                'seed': seed_num,
                                'team_id': team.get('id', ''),
                                'team_name': team.get('name', ''),
                                'record': team.get('record', '')
                            })

                if teams:
                    df = pd.DataFrame(teams)
                    df.to_csv('bracketology.csv', index=False)
                    print(f"Saved {len(df)} bracketology teams to bracketology.csv")
            except Exception as e:
                print(f"Error processing bracketology data: {e}")

            return data
        else:
            print(f"Bracketology data not available (status code: {response.status_code})")
            return None
    except Exception as e:
        print(f"Error fetching bracketology: {e}")
        return None

# Run the functions to collect and save the data
if __name__ == "__main__":
    print("Starting NCAA basketball data collection...")

    # Get team info first
    teams_df = get_teams_to_csv()

    # Get rankings
    get_rankings_to_csv()

    # Get bracketology
    get_bracketology_to_json()

    # Get team stats (this will take the longest)
    if not teams_df.empty:
        get_all_team_stats(teams_df)

    print("\nData collection complete! All data has been saved to CSV and JSON files.")
