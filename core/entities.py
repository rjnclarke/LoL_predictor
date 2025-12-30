class Player:
    """Simple player object storing only identity for now."""
    def __init__(self, puuid):
        self.puuid = puuid


class Team:
    """Container assigning the five fixed roles to Player objects."""
    ROLES_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

    def __init__(self, players):
        """
        players : list[Player]
            Expected order: TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
        """
        if len(players) != 5:
            raise ValueError("Team must receive exactly five Player objects.")
        self.top, self.jungle, self.middle, self.bottom, self.utility = players
        self.players = players  # keep list for easy iteration

    @property
    def puuids(self):
        return [p.puuid for p in self.players]

    def __iter__(self):
        return iter(self.players)


class Match:
    """Simply groups two Team instances for visual comparison."""
    def __init__(self, blue_team, red_team):
        self.blue = blue_team
        self.red = red_team