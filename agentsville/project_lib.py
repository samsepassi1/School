"""
project_lib.py
==============

Helper library for the AgentsVille Trip Planner project.

Contents
--------
* Pydantic models used throughout the project (``Traveler``, ``Activity``,
  ``Weather``, ``ItineraryActivity``, ``ItineraryDay``, ``TravelPlan``).
* A small mock "external API" returning weather forecasts and activities for
  the city of AgentsVille between 2025-06-09 and 2025-06-20.
* A handful of pure helper utilities used by the agents and tools defined
  in ``project_starter.ipynb``.

The notebook expects to import the public names defined at the bottom of this
module via ``from project_lib import *``.
"""

from __future__ import annotations

import math
import operator
from datetime import date, timedelta
from enum import Enum
from typing import Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CITY = "AgentsVille"


class WeatherCondition(str, Enum):
    SUNNY = "sunny"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    WINDY = "windy"


# ---------------------------------------------------------------------------
# Pydantic models (the notebook also defines ``VacationInfo`` itself).
# ---------------------------------------------------------------------------

class Traveler(BaseModel):
    """A single person on the trip."""

    name: str
    age: int = Field(..., ge=0, le=120)
    interests: List[str] = Field(
        default_factory=list,
        description="Free-form interest tags, e.g. ['food', 'history', 'music'].",
    )


class Weather(BaseModel):
    """A simple daily weather forecast."""

    date: date
    condition: WeatherCondition
    temperature_high_c: float
    temperature_low_c: float
    precipitation_chance: float = Field(..., ge=0.0, le=1.0)
    description: str


class Activity(BaseModel):
    """An activity offered in AgentsVille on a specific date."""

    activity_id: str = Field(..., description="Stable identifier, e.g. 'ACT-007'.")
    name: str
    date: date
    start_time: str = Field(..., description="24h HH:MM, e.g. '09:30'.")
    end_time: str = Field(..., description="24h HH:MM, e.g. '11:00'.")
    location: str
    description: str
    price_usd: float = Field(..., ge=0.0)
    related_interests: List[str] = Field(default_factory=list)
    is_outdoor: bool = Field(
        ...,
        description=(
            "True when the activity is primarily outdoors and might be impacted "
            "by adverse weather (rain, storms, extreme heat, etc.)."
        ),
    )


class ItineraryActivity(BaseModel):
    """A single activity slotted into the itinerary on a specific day."""

    activity_id: str
    name: str
    start_time: str
    end_time: str
    location: str
    description: str
    price_usd: float = Field(..., ge=0.0)


class ItineraryDay(BaseModel):
    date: date
    weather_summary: str
    activities: List[ItineraryActivity]


class TravelPlan(BaseModel):
    """The structured output produced by the ItineraryAgent."""

    city: str
    start_date: date
    end_date: date
    travelers: List[str] = Field(
        ..., description="Names of the people on the trip."
    )
    days: List[ItineraryDay]
    total_cost_usd: float = Field(..., ge=0.0)
    notes: str = ""

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: date, info):  # type: ignore[override]
        start = info.data.get("start_date")
        if start is not None and v < start:
            raise ValueError("end_date must be on or after start_date")
        return v


# ---------------------------------------------------------------------------
# Mock "external API" data
# ---------------------------------------------------------------------------

# Weather forecast for AgentsVille — covers 2025-06-09 through 2025-06-20.
_WEATHER_RECORDS: List[Weather] = [
    Weather(
        date=date(2025, 6, 9),
        condition=WeatherCondition.SUNNY,
        temperature_high_c=27.0,
        temperature_low_c=18.0,
        precipitation_chance=0.05,
        description="Clear and warm, light breeze through the afternoon.",
    ),
    Weather(
        date=date(2025, 6, 10),
        condition=WeatherCondition.SUNNY,
        temperature_high_c=28.0,
        temperature_low_c=18.0,
        precipitation_chance=0.05,
        description="Bright and sunny all day, perfect for being outdoors.",
    ),
    Weather(
        date=date(2025, 6, 11),
        condition=WeatherCondition.PARTLY_CLOUDY,
        temperature_high_c=26.0,
        temperature_low_c=17.0,
        precipitation_chance=0.20,
        description="Mix of sun and clouds, slight chance of an afternoon shower.",
    ),
    Weather(
        date=date(2025, 6, 12),
        condition=WeatherCondition.CLOUDY,
        temperature_high_c=23.0,
        temperature_low_c=16.0,
        precipitation_chance=0.30,
        description="Overcast and cool, mostly dry but humid.",
    ),
    Weather(
        date=date(2025, 6, 13),
        condition=WeatherCondition.RAINY,
        temperature_high_c=20.0,
        temperature_low_c=15.0,
        precipitation_chance=0.85,
        description="Steady rain throughout the day, strong winds at times.",
    ),
    Weather(
        date=date(2025, 6, 14),
        condition=WeatherCondition.SUNNY,
        temperature_high_c=29.0,
        temperature_low_c=19.0,
        precipitation_chance=0.05,
        description="Hot and sunny, great beach and lake weather.",
    ),
    Weather(
        date=date(2025, 6, 15),
        condition=WeatherCondition.STORMY,
        temperature_high_c=22.0,
        temperature_low_c=17.0,
        precipitation_chance=0.95,
        description="Thunderstorms expected — outdoor activities should be avoided.",
    ),
    Weather(
        date=date(2025, 6, 16),
        condition=WeatherCondition.PARTLY_CLOUDY,
        temperature_high_c=25.0,
        temperature_low_c=17.0,
        precipitation_chance=0.15,
        description="Pleasant, mostly sunny, brief cloudy spells.",
    ),
    Weather(
        date=date(2025, 6, 17),
        condition=WeatherCondition.SUNNY,
        temperature_high_c=27.0,
        temperature_low_c=18.0,
        precipitation_chance=0.05,
        description="Clear skies and warm temperatures.",
    ),
    Weather(
        date=date(2025, 6, 18),
        condition=WeatherCondition.WINDY,
        temperature_high_c=24.0,
        temperature_low_c=16.0,
        precipitation_chance=0.10,
        description="Strong winds across the city, good for sailing, tough for cycling.",
    ),
    Weather(
        date=date(2025, 6, 19),
        condition=WeatherCondition.CLOUDY,
        temperature_high_c=22.0,
        temperature_low_c=15.0,
        precipitation_chance=0.25,
        description="Grey and cool, occasional drizzle possible.",
    ),
    Weather(
        date=date(2025, 6, 20),
        condition=WeatherCondition.SUNNY,
        temperature_high_c=28.0,
        temperature_low_c=18.0,
        precipitation_chance=0.05,
        description="Sunny and warm, ideal for outdoor exploration.",
    ),
]


# Five activities per day across a wide variety of interests + indoor/outdoor mix.
def _build_activities() -> List[Activity]:
    raw = [
        # 2025-06-10 — sunny
        ("ACT-001", "Riverside Park Walking Tour", date(2025, 6, 10), "09:00", "11:00",
         "Riverside Park", "A guided walking tour through AgentsVille's scenic riverside park.",
         25.0, ["outdoor", "history", "walking"], True),
        ("ACT-002", "AgentsVille Art Museum", date(2025, 6, 10), "11:30", "14:00",
         "Downtown — Museum District", "Explore three floors of contemporary and classical art.",
         30.0, ["art", "culture"], False),
        ("ACT-003", "Chef Marco's Italian Cooking Class", date(2025, 6, 10), "15:00", "17:30",
         "Old Town Kitchen Studio", "Hands-on class making fresh pasta and tiramisu.",
         85.0, ["food", "cooking"], False),
        ("ACT-004", "Sunset Jazz Cruise", date(2025, 6, 10), "19:00", "21:30",
         "AgentsVille Marina", "Live jazz quartet aboard a sunset boat cruise with light tapas.",
         95.0, ["music", "food", "outdoor"], True),
        ("ACT-005", "Night Market Food Tour", date(2025, 6, 10), "20:00", "22:00",
         "Old Town Square", "Walking tour of the night market with eight tasting stops.",
         55.0, ["food", "outdoor", "culture"], True),

        # 2025-06-11 — partly cloudy
        ("ACT-006", "Mountain Bike Adventure", date(2025, 6, 11), "08:30", "12:00",
         "AgentsVille Mountain Trails", "Guided mountain bike ride along intermediate trails.",
         70.0, ["outdoor", "adventure", "sports"], True),
        ("ACT-007", "Old Town History Walking Tour", date(2025, 6, 11), "10:00", "12:00",
         "Old Town", "Two-hour tour of cobblestone streets with a local historian.",
         20.0, ["history", "walking", "culture"], True),
        ("ACT-008", "Brewery Hop Tour", date(2025, 6, 11), "14:00", "17:00",
         "Brewery District", "Visit three craft breweries with tastings and brewer Q&A.",
         65.0, ["food", "drinks"], False),
        ("ACT-009", "Skyline Observatory Visit", date(2025, 6, 11), "17:30", "19:00",
         "AgentsVille Tower, 88th Floor", "Panoramic city views from the highest observatory in town.",
         40.0, ["views", "architecture"], False),
        ("ACT-010", "Live Comedy Night", date(2025, 6, 11), "20:00", "22:00",
         "Laugh Lounge", "Stand-up showcase featuring four headliners.",
         30.0, ["entertainment", "nightlife"], False),

        # 2025-06-12 — cloudy
        ("ACT-011", "Modern Art Gallery Tour", date(2025, 6, 12), "10:00", "12:00",
         "Riverwalk Gallery", "Curated tour of the city's most-talked-about modern gallery.",
         25.0, ["art", "culture"], False),
        ("ACT-012", "Underground Speakeasy Cocktail Class", date(2025, 6, 12), "13:00", "15:00",
         "Hidden Cellar Bar", "Hands-on cocktail-making in a 1920s-themed speakeasy.",
         60.0, ["drinks", "history"], False),
        ("ACT-013", "Cooking with Locals: Street Food Edition", date(2025, 6, 12), "15:30", "18:00",
         "Market Hall Kitchen", "Cook three iconic AgentsVille street-food dishes with a local chef.",
         75.0, ["food", "cooking", "culture"], False),
        ("ACT-014", "Vintage Vinyl & Record Shop Walk", date(2025, 6, 12), "16:00", "18:00",
         "Music Quarter", "Walking tour through the city's most beloved record shops.",
         15.0, ["music", "shopping"], True),
        ("ACT-015", "Ghost Tour at Night", date(2025, 6, 12), "20:30", "22:30",
         "Old Town", "Spooky storytelling tour through AgentsVille's haunted alleys.",
         35.0, ["history", "entertainment", "outdoor"], True),

        # 2025-06-13 — rainy
        ("ACT-016", "Indoor Climbing Gym Session", date(2025, 6, 13), "09:00", "11:00",
         "Vertical AgentsVille", "Drop-in session at the city's largest indoor climbing gym.",
         30.0, ["sports", "adventure"], False),
        ("ACT-017", "Heritage Theater Cinema Marathon", date(2025, 6, 13), "11:30", "16:00",
         "Heritage Theater", "Triple feature of restored classic films with intermissions.",
         25.0, ["entertainment", "film"], False),
        ("ACT-018", "Pottery Workshop", date(2025, 6, 13), "13:00", "16:00",
         "Clay Studio", "Wheel-throwing pottery workshop, take home what you make.",
         70.0, ["art", "crafts"], False),
        ("ACT-019", "Jazz Club Night at Blue Note 2", date(2025, 6, 13), "19:30", "22:30",
         "Downtown Jazz Club", "Two sets from a local jazz quintet, dinner available.",
         50.0, ["music", "nightlife"], False),
        ("ACT-020", "Spa & Wellness Evening", date(2025, 6, 13), "17:00", "20:00",
         "Lotus Spa", "Three-hour spa circuit with sauna, steam, and a 30-minute massage.",
         120.0, ["wellness", "relaxation"], False),

        # 2025-06-14 — sunny
        ("ACT-021", "Lakeside Kayaking", date(2025, 6, 14), "09:00", "11:30",
         "Lake AgentsVille", "Guided kayaking around the calm bays of Lake AgentsVille.",
         55.0, ["outdoor", "adventure", "sports"], True),
        ("ACT-022", "Botanical Gardens Tour", date(2025, 6, 14), "11:00", "13:00",
         "AgentsVille Botanical Gardens", "Stroll through themed gardens at peak summer bloom.",
         18.0, ["nature", "outdoor"], True),
        ("ACT-023", "Outdoor Food Festival", date(2025, 6, 14), "12:00", "16:00",
         "Festival Park", "Annual food festival with 40+ vendors and live music stages.",
         20.0, ["food", "outdoor", "music"], True),
        ("ACT-024", "Open-Air Concert in the Park", date(2025, 6, 14), "18:00", "21:00",
         "Festival Park Bandshell", "Headline indie band performs an open-air evening concert.",
         60.0, ["music", "outdoor"], True),
        ("ACT-025", "Stargazing Picnic", date(2025, 6, 14), "21:30", "23:30",
         "Hilltop Overlook", "Astronomer-led stargazing with telescopes and a picnic basket.",
         45.0, ["nature", "outdoor", "science"], True),

        # 2025-06-15 — stormy
        ("ACT-026", "AgentsVille Science Museum", date(2025, 6, 15), "09:30", "12:30",
         "Museum District", "Hands-on exhibits across robotics, space, and biology.",
         28.0, ["science", "family"], False),
        ("ACT-027", "Indoor Aquarium Visit", date(2025, 6, 15), "11:00", "13:30",
         "Harbour Aquarium", "Massive indoor aquarium with kelp forest and shark tunnel.",
         32.0, ["nature", "family"], False),
        ("ACT-028", "Wine Tasting at Cellar 9", date(2025, 6, 15), "14:00", "16:00",
         "Wine District — Cellar 9", "Guided tasting of nine regional wines with cheese pairings.",
         55.0, ["drinks", "food"], False),
        ("ACT-029", "Library Cafe Reading Club", date(2025, 6, 15), "15:00", "17:00",
         "Central Library Cafe", "Low-key reading club discussion with coffee and pastries.",
         10.0, ["books", "relaxation"], False),
        ("ACT-030", "Symphony Orchestra Concert", date(2025, 6, 15), "19:30", "21:30",
         "Concert Hall", "AgentsVille Symphony performs a classical romantic-era program.",
         70.0, ["music", "culture"], False),

        # 2025-06-16 — partly cloudy
        ("ACT-031", "Bicycle City Loop", date(2025, 6, 16), "09:00", "12:00",
         "Bike Hub Downtown", "Three-hour guided bike loop covering the city's main sights.",
         40.0, ["outdoor", "sports", "history"], True),
        ("ACT-032", "Photography Walk", date(2025, 6, 16), "10:00", "12:30",
         "Old Town", "Photo-focused walking tour with a professional photographer.",
         45.0, ["art", "walking", "outdoor"], True),
        ("ACT-033", "Lunchtime Cooking Demo", date(2025, 6, 16), "12:30", "14:00",
         "Market Hall Kitchen", "Live cooking demo with tasting menu of seasonal dishes.",
         35.0, ["food", "cooking"], False),
        ("ACT-034", "Vintage Shopping Tour", date(2025, 6, 16), "15:00", "17:30",
         "Vintage Quarter", "Curated tour of the city's best vintage and second-hand shops.",
         20.0, ["shopping", "fashion"], False),
        ("ACT-035", "Rooftop Bar Hop", date(2025, 6, 16), "20:00", "23:00",
         "Downtown Skyline", "Visit three rooftop bars with skyline views.",
         70.0, ["drinks", "nightlife", "views"], False),

        # 2025-06-17 — sunny
        ("ACT-036", "Hot Air Balloon Sunrise Flight", date(2025, 6, 17), "05:30", "08:00",
         "Balloon Field North", "Sunrise hot-air balloon flight over the AgentsVille valley.",
         220.0, ["adventure", "outdoor", "views"], True),
        ("ACT-037", "Farmers Market Tasting Tour", date(2025, 6, 17), "09:30", "11:30",
         "Saturday Farmers Market", "Tasting tour through the city's famous Saturday farmers market.",
         40.0, ["food", "outdoor"], True),
        ("ACT-038", "Historic Castle Day Trip", date(2025, 6, 17), "10:00", "16:00",
         "AgentsVille Castle", "Half-day excursion to the medieval castle on the city's edge.",
         60.0, ["history", "outdoor"], True),
        ("ACT-039", "Afternoon Tea at the Grand", date(2025, 6, 17), "15:00", "16:30",
         "The Grand Hotel", "Classic afternoon tea service with pastries and finger sandwiches.",
         55.0, ["food"], False),
        ("ACT-040", "Theater Performance: 'AgentsVille Nights'", date(2025, 6, 17), "19:30", "22:00",
         "Royal Theatre", "Award-winning new play about life in modern AgentsVille.",
         85.0, ["entertainment", "culture"], False),

        # 2025-06-18 — windy
        ("ACT-041", "Sailing Lesson on the Bay", date(2025, 6, 18), "09:00", "12:00",
         "Marina Sail School", "Beginner sailing lesson on the wind-friendly bay.",
         110.0, ["adventure", "outdoor", "sports"], True),
        ("ACT-042", "Maritime Museum Tour", date(2025, 6, 18), "13:00", "15:00",
         "Harbour Museum", "Two centuries of AgentsVille's seafaring history.",
         22.0, ["history", "culture"], False),
        ("ACT-043", "Chocolate-Making Workshop", date(2025, 6, 18), "15:30", "17:30",
         "Cocoa Lab", "Make and take home a small box of artisan chocolates.",
         65.0, ["food", "crafts"], False),
        ("ACT-044", "Comedy Improv Show", date(2025, 6, 18), "20:00", "21:30",
         "Improv Theatre", "Audience-driven improv comedy show.",
         28.0, ["entertainment"], False),
        ("ACT-045", "Late-Night Dessert Tour", date(2025, 6, 18), "21:00", "23:00",
         "Sweet Quarter", "Walking tour with stops at five late-night dessert spots.",
         40.0, ["food", "walking"], True),

        # 2025-06-19 — cloudy
        ("ACT-046", "AgentsVille History Museum", date(2025, 6, 19), "10:00", "12:30",
         "Museum District", "Permanent exhibits on the city's two-thousand-year history.",
         18.0, ["history", "culture"], False),
        ("ACT-047", "Indoor Botanical Conservatory", date(2025, 6, 19), "11:00", "13:00",
         "Glasshouse Conservatory", "Tropical glasshouse with rare orchids and butterflies.",
         15.0, ["nature"], False),
        ("ACT-048", "Coffee & Pastry Crawl", date(2025, 6, 19), "13:30", "16:00",
         "Cafe District", "Walk between five top-rated cafes with curated tastings.",
         35.0, ["food", "walking"], True),
        ("ACT-049", "Live Music: Acoustic Lounge", date(2025, 6, 19), "19:00", "21:00",
         "Acoustic Lounge", "Singer-songwriter showcase with three local artists.",
         25.0, ["music"], False),
        ("ACT-050", "Late Bookstore & Wine Evening", date(2025, 6, 19), "20:00", "22:30",
         "Inkwell Bookstore", "Independent bookstore opens late for wine, talks, and signings.",
         20.0, ["books", "drinks"], False),

        # 2025-06-20 — sunny
        ("ACT-051", "Sunrise Yoga in the Park", date(2025, 6, 20), "07:00", "08:30",
         "Riverside Park Lawn", "Drop-in sunrise yoga class.",
         18.0, ["wellness", "outdoor"], True),
        ("ACT-052", "Boat Cruise Along the Coast", date(2025, 6, 20), "10:00", "13:00",
         "AgentsVille Marina", "Three-hour coastal cruise with lunch on board.",
         95.0, ["outdoor", "food", "views"], True),
        ("ACT-053", "Architectural Walking Tour", date(2025, 6, 20), "11:00", "13:00",
         "Downtown", "Tour of AgentsVille's most striking modern and historic buildings.",
         25.0, ["history", "architecture", "walking"], True),
        ("ACT-054", "Family-Style Italian Dinner", date(2025, 6, 20), "18:30", "21:00",
         "Trattoria della Piazza", "Multi-course family-style dinner at a beloved trattoria.",
         70.0, ["food"], False),
        ("ACT-055", "Fireworks Finale by the River", date(2025, 6, 20), "21:30", "22:30",
         "Riverside Esplanade", "Free public fireworks display along the river.",
         0.0, ["outdoor", "entertainment"], True),
    ]

    return [
        Activity(
            activity_id=row[0],
            name=row[1],
            date=row[2],
            start_time=row[3],
            end_time=row[4],
            location=row[5],
            description=row[6],
            price_usd=row[7],
            related_interests=row[8],
            is_outdoor=row[9],
        )
        for row in raw
    ]


_ACTIVITY_RECORDS: List[Activity] = _build_activities()


# ---------------------------------------------------------------------------
# Mock API helpers
# ---------------------------------------------------------------------------

def _date_range(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def get_weather_data(start_date: date, end_date: date) -> List[Weather]:
    """Return the daily weather forecast for AgentsVille between two dates (inclusive)."""

    return [w for w in _WEATHER_RECORDS if start_date <= w.date <= end_date]


def get_activities_data(start_date: date, end_date: date) -> List[Activity]:
    """Return all activities offered in AgentsVille between two dates (inclusive)."""

    return [a for a in _ACTIVITY_RECORDS if start_date <= a.date <= end_date]


def get_activities_by_date(target_date: date) -> List[Activity]:
    """Return every activity offered on a given date."""

    return [a for a in _ACTIVITY_RECORDS if a.date == target_date]


# ---------------------------------------------------------------------------
# Calculator helper used by ``calculator_tool``
# ---------------------------------------------------------------------------

_ALLOWED_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "**": operator.pow,
}


def safe_calculator(expression: str) -> float:
    """Evaluate a simple arithmetic expression safely.

    Supports ``+``, ``-``, ``*``, ``/``, ``%``, ``**`` and parentheses on
    floating-point numbers. Anything else raises ``ValueError``.
    """

    import ast

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Num,  # py<3.8 compat
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
        ast.Load,
    )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse expression: {expression!r}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(
                f"Unsupported syntax in expression: {type(node).__name__}"
            )

    return float(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {}))


__all__ = [
    "CITY",
    "WeatherCondition",
    "Traveler",
    "Weather",
    "Activity",
    "ItineraryActivity",
    "ItineraryDay",
    "TravelPlan",
    "get_weather_data",
    "get_activities_data",
    "get_activities_by_date",
    "safe_calculator",
]
