"""Export NHL shot attempts. Requires Python 3.10+ and system curl."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlencode


SHOT_TYPES = {"goal", "shot-on-goal", "missed-shot", "blocked-shot"}
# curl transport/HTTP failures only; local/configuration errors still terminate.
DOWNLOAD_ERRORS = {5, 6, 7, 16, 18, 22, 28, 35, 52, 55, 56, 60, 92, 95, 96, 97}
FIELDS = """season game_id game_type game_date start_time_utc venue game_state
home_team_id home_team_abbrev away_team_id away_team_abbrev
event_id event_type sort_order period period_type time_in_period time_remaining
is_shootout situationCode home_team_defending_side event_owner_team_id
shooter_id shooter_name goalie_id goalie_name blocker_id blocker_name
shot_type reason x_coord y_coord zone_code""".split()


def fetch(url, cache, failures, refresh=False):
    if cache.exists() and not refresh:
        return json.loads(gzip.decompress(cache.read_bytes()))
    time.sleep(0.2)
    response = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--location",
         "--connect-timeout", "10", "--max-time", "30", url],
        capture_output=True,
    )
    try:
        response.check_returncode()
    except subprocess.CalledProcessError as error:
        if error.returncode not in DOWNLOAD_ERRORS:
            raise
        message = f"{error}: {response.stderr.decode().strip()}"
        print(f"Download failed: {url}\n{message}", file=sys.stderr, flush=True)
        failures.append({"url": url, "exception": message})
        return None
    data = json.loads(response.stdout)
    cache.write_bytes(gzip.compress(response.stdout, mtime=0))
    return data


def shot_rows(game):
    players = {
        p["playerId"]: " ".join(
            p.get(k, {}).get("default", "") for k in ("firstName", "lastName")
        ).strip()
        for p in game.get("rosterSpots", [])
    }
    common = {
        "season": game["season"], "game_id": game["id"],
        "game_type": game["gameType"], "game_date": game["gameDate"],
        "start_time_utc": game.get("startTimeUTC"),
        "venue": game.get("venue", {}).get("default"), "game_state": game["gameState"],
    }
    for side in ("home", "away"):
        team = game[f"{side}Team"]
        common.update({f"{side}_team_id": team["id"], f"{side}_team_abbrev": team["abbrev"]})
    rows = []
    for play in game["plays"]:
        if play["typeDescKey"] not in SHOT_TYPES:
            continue
        details = play.get("details", {})
        period = play["periodDescriptor"]
        row = dict(common, event_id=play["eventId"], event_type=play["typeDescKey"],
                   sort_order=play["sortOrder"], period=period["number"],
                   period_type=period["periodType"], time_in_period=play.get("timeInPeriod"),
                   time_remaining=play.get("timeRemaining"),
                   is_shootout=int(period["periodType"] == "SO"),
                   situationCode=play.get("situationCode"),
                   home_team_defending_side=play.get("homeTeamDefendingSide"))
        for column, source in {
            "event_owner_team_id": "eventOwnerTeamId", "shot_type": "shotType",
            "reason": "reason", "x_coord": "xCoord", "y_coord": "yCoord", "zone_code": "zoneCode",
        }.items():
            row[column] = details.get(source)
        for role, source in {
            "shooter": "scoringPlayerId" if play["typeDescKey"] == "goal" else "shootingPlayerId",
            "goalie": "goalieInNetId", "blocker": "blockingPlayerId",
        }.items():
            player_id = details.get(source)
            row[f"{role}_id"] = player_id
            row[f"{role}_name"] = players.get(player_id)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=20252026)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data",
                        help="Data root containing shots/, coverage/, and raw/")
    parser.add_argument("--refresh", action="store_true", help="Refetch cached API responses")
    args = parser.parse_args()
    output = args.output_dir
    (output / "shots").mkdir(parents=True, exist_ok=True)
    (output / "coverage").mkdir(parents=True, exist_ok=True)
    cache = output / "raw" / str(args.season)
    cache.mkdir(parents=True, exist_ok=True)
    report = {"season": args.season, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "in_progress", "download_failures": [], "games": []}
    report_path = output / "coverage" / f"coverage_{args.season}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    url = "https://api.nhle.com/stats/rest/en/game?" + urlencode({
        "limit": -1, "cayenneExp": f"season={args.season} and (gameType=2 or gameType=3)"})
    report["discovery_url"] = url
    discovery = fetch(url, cache / "games.json.gz", report["download_failures"], args.refresh)
    if discovery is None:
        report["status"] = "discovery_failed"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("Game discovery failed; no CSV exported.", file=sys.stderr)
        return
    assert len(discovery["data"]) == discovery["total"], "Incomplete game discovery"
    games = sorted((g for g in discovery["data"] if g["gameStateId"] == 7),
                   key=lambda g: (g["gameDate"], g["id"]))
    assert games, "No completed games found"
    assert len({g["id"] for g in games}) == len(games), "Duplicate game IDs"
    report.update(discovered_completed_games=len(games),
                  excluded_unplayed_game_ids=[g["id"] for g in discovery["data"] if g["gameStateId"] != 7])
    rows, seen = [], set()
    for index, scheduled in enumerate(games, 1):
        game_id = scheduled["id"]
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        game = fetch(url, cache / f"{game_id}.json.gz", report["download_failures"], args.refresh)
        entry = {"game_id": game_id, "game_type": scheduled["gameType"], "source_url": url}
        report["games"].append(entry)
        if game is None:
            entry["status"] = "download_failed"
            continue
        assert game["id"] == game_id and game["season"] == args.season
        assert game["gameType"] in (2, 3) and game["gameState"] in ("OFF", "FINAL")
        extracted = shot_rows(game)
        for row in extracted:
            key = (game_id, row["event_id"])
            assert key not in seen, f"Duplicate event: {key}"
            seen.add(key)
        source_counts = Counter(p["typeDescKey"] for p in game["plays"] if p["typeDescKey"] in SHOT_TYPES)
        assert Counter(r["event_type"] for r in extracted) == source_counts
        sog = Counter(r["event_owner_team_id"] for r in extracted
                      if not r["is_shootout"] and r["event_type"] in ("goal", "shot-on-goal"))
        checks = [{"team_id": game[side]["id"], "reported": game[side].get("sog"),
                   "extracted": sog[game[side]["id"]]} for side in ("homeTeam", "awayTeam")]
        entry.update(status="exported" if extracted else "missing_shot_events", rows=len(extracted),
                     event_counts=dict(source_counts),
                     missing_coordinate_event_ids=[r["event_id"] for r in extracted
                         if r["x_coord"] is None or r["y_coord"] is None],
                     out_of_bounds_coordinate_event_ids=[r["event_id"] for r in extracted
                         if (r["x_coord"] is not None and abs(r["x_coord"]) > 100)
                         or (r["y_coord"] is not None and abs(r["y_coord"]) > 42.5)],
                     sog_checks=checks)
        rows.extend(extracted)
        if index % 50 == 0 or index == len(games):
            print(f"Processed {index}/{len(games)} games; {len(rows):,} shot attempts", flush=True)
    rows.sort(key=lambda r: (r["game_date"], r["game_id"], r["sort_order"], r["event_id"]))
    csv_path = output / "shots" / f"shots_{args.season}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == len(rows) == len(seen)
    assert len({(r["game_id"], r["event_id"]) for r in saved}) == len(rows)
    exported = [g for g in report["games"] if g["status"] == "exported"]
    report.update(status="complete" if len(exported) == len(games) else "partial",
                  exported_games=len(exported), rows=len(rows), event_counts=dict(Counter(r["event_type"] for r in rows)),
                  shootout_rows=sum(r["is_shootout"] for r in rows),
                  missing_coordinate_rows=sum(len(g.get("missing_coordinate_event_ids", [])) for g in report["games"]),
                  out_of_bounds_coordinate_rows=sum(len(g.get("out_of_bounds_coordinate_event_ids", [])) for g in report["games"]),
                  sog_discrepancies=[dict(game_id=g["game_id"], **check) for g in exported
                      for check in g["sog_checks"] if check["reported"] != check["extracted"]])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}: {len(rows):,} rows from {len(exported)}/{len(games)} games → {csv_path}")


if __name__ == "__main__":
    main()
