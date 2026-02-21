import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.oauth2 import service_account
from google.genai import types

# 1. Path to your Service Account JSON file
BASE_PATH = Path(__file__).resolve().parent
SERVICE_ACCOUNT_FILE = str(
    BASE_PATH.parent / "parfas_apis" / "gemini_api" / "gen-lang-client-0203982997-2afd9d9aeb98.json"
)
PROJECT_ID = "gen-lang-client-0203982997"
LOCATION = "global"

# 2. Define the structure for event data
class EventInfo(BaseModel):
    """Schema for a single event"""
    event_name: str = Field(description="Name of the event")
    event_type: str = Field(description="Category: sports, concert, festival, conference, etc.")
    sport_type: Optional[str] = Field(default=None, description="If sports event: football, cricket, tennis, golf, rugby, etc.")
    date: str = Field(description="Event date in DD/MM/YYYY format")
    time: Optional[str] = Field(default=None, description="Event start time if known")
    venue_name: str = Field(description="Name of the venue")
    city: str = Field(description="City where event takes place")
    region: str = Field(description="Region/County in UK")
    postcode: Optional[str] = Field(default=None, description="Venue postcode if known")
    expected_attendance: Optional[str] = Field(default=None, description="Estimated crowd size: small/medium/large/very large")
    description: Optional[str] = Field(default=None, description="Brief description of the event")
    source_reference: Optional[str] = Field(default=None, description="Typical source website for this type of event")

class EventsResponse(BaseModel):
    """Schema for the full response containing multiple events"""
    events: List[EventInfo] = Field(description="List of all major events")
    total_events: int = Field(description="Total number of events found")


# 3. Load Credentials and Initialize Client
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=SCOPES
)

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=creds
)


def get_uk_events(start_date: str, num_days: int = 7) -> EventsResponse:
    """
    Fetch major UK events for a given date range.
    
    Args:
        start_date: Start date in DD/MM/YYYY format
        num_days: Number of days to look ahead (default 7)
    
    Returns:
        EventsResponse with list of events
    """
    
    prompt = f"""
You are a UK events researcher. Provide a comprehensive list of major events happening across the United Kingdom.

**Date Range:** {start_date} to {num_days} days ahead

**Event Categories to Include:**
1. **Sports Events:**
   - Football: Premier League, Championship, FA Cup, League Cup matches (check premierleague.com, efl.com)
   - Rugby: Six Nations, Premiership Rugby matches (check englandrugby.com, premiershiprugby.com)
   - Cricket: Any domestic or international matches (check ecb.co.uk)
   - Tennis: ATP/WTA tournaments (check lta.org.uk)
   - Golf: PGA Tour, European Tour events (check europeantour.com)
   - Horse Racing: Major race meetings (check britishhorseracing.com)
   - Boxing/MMA: Major fights
   - Snooker/Darts: Ranking events

2. **Music & Entertainment:**
   - Major concerts and arena tours (check ticketmaster.co.uk, axs.com)
   - Theatre productions in West End and major venues
   - Comedy shows at large venues

3. **Festivals & Cultural Events:**
   - Music festivals
   - Food and drink festivals
   - Cultural celebrations
   - Religious/community events

4. **Conferences & Exhibitions:**
   - Major industry conferences
   - Trade shows at ExCeL London, NEC Birmingham, etc.
   - Academic conferences

5. **Other Large Gatherings:**
   - Political rallies or demonstrations
   - Marathons and large sporting participation events
   - Award ceremonies

**Instructions:**
- Focus on events expecting 5,000+ attendees
- Cover ALL regions of the UK: London, South East, South West, Midlands, North West, North East, Yorkshire, Scotland, Wales, Northern Ireland
- Include the venue's city and postcode where possible
- Estimate attendance as: small (<5,000), medium (5,000-20,000), large (20,000-50,000), very large (50,000+)
- Reference typical source websites for each event type
- Include only large and very large events in the final output.

**Important:** Provide as many events as possible across different categories and regions. Be thorough and comprehensive.
"""
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EventsResponse,
            # Wrap thinking_level inside thinking_config
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,        # Optional: set to True if you want to see the "chain of thought"
            )
        )
    )

    return response.parsed


def display_events(events_response: EventsResponse):
    """Pretty print the events"""
    print(f"\n{'='*80}")
    print(f"MAJOR UK EVENTS - Total Found: {events_response.total_events}")
    print(f"{'='*80}\n")
    
    # Group by event type
    events_by_type = {}
    for event in events_response.events:
        event_type = event.event_type
        if event_type not in events_by_type:
            events_by_type[event_type] = []
        events_by_type[event_type].append(event)
    
    for event_type, events in events_by_type.items():
        print(f"\n{'─'*40}")
        print(f"📌 {event_type.upper()} ({len(events)} events)")
        print(f"{'─'*40}")
        
        for event in events:
            print(f"\n  🎯 {event.event_name}")
            print(f"     📅 Date: {event.date} {f'at {event.time}' if event.time else ''}")
            print(f"     📍 Venue: {event.venue_name}")
            print(f"     🏙️  Location: {event.city}, {event.region}")
            if event.postcode:
                print(f"     📮 Postcode: {event.postcode}")
            if event.sport_type:
                print(f"     ⚽ Sport: {event.sport_type}")
            if event.expected_attendance:
                print(f"     👥 Expected Attendance: {event.expected_attendance}")
            if event.description:
                print(f"     📝 {event.description}")


if __name__ == "__main__":
    # Get events for the next 7 days starting from today
    START_DATE = "28/01/2026"
    NUM_DAYS = 7
    
    print(f"Fetching major UK events from {START_DATE} for {NUM_DAYS} days...")
    print("(Note: Results are based on model knowledge, not real-time data)")
    
    result = get_uk_events(START_DATE, NUM_DAYS)
    
    # Display formatted output
    display_events(result)
    
    # Also save raw JSON
    print(f"\n{'='*80}")
    print("RAW JSON OUTPUT:")
    print(f"{'='*80}")
    print(result.model_dump_json(indent=2))
