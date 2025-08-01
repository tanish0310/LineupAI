from models.schema import Session, GameweekStats

session = Session()

latest_stats = session.query(GameweekStats).filter(
    GameweekStats.gameweek_id < 39
).order_by(GameweekStats.gameweek_id.desc()).limit(5).all()

for stat in latest_stats:
    print(f"Player {stat.player_id} | GW {stat.gameweek_id} | Points: {stat.points}")
