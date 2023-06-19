
# Surfline Sessions Project

At Surfline we are always looking for ways to better understand where, when and why people are surfing and how we can use that information to improve our products and services. One way to explore this is by looking at the data we collect from our users while using our Sessions app. This data is collected from the user's smartwatch and includes GPS location, speed, and other information about their Session and the waves they've caught.

Here, we specifically are interested in visualizations of the data and how these visualizations could either
- be used to help us (internally) understand the data better and improve our products and services
- be made directly into a product for users to be explored in the app / on web

For example, 
Tidal variance is very large for in the UK. Do people take off at different locations depending on the tide level?
Could we help them make a decision on where to sit based on the tide level by showing them a visualization? How would such a map looks like?

## Data

This repository contains sample data from Surfline Sessions between October 2019 and March 2023 for Bantham, UK. The data is anonymized and does not contain any personally identifiable information.
We provide Sessions data as well as forecast data for the time of the Sessions, such as breaking wave heights, tides, and Surfline's Ratings scale for the time of the Sessions.

Forecast data is contained in `BanthamForecast.csv`, with each row for one Session.
GPS data is contained in a JSON file, with the Session ID in the file name, and each event (e.g. a surfed wave) as a nested object.

### Forecast Data

| Property            | Type              | Description                                       |
|---------------------|------------------|---------------------------------------------------|
| id | String | Session object ID. |
| startTimestamp | Integer  | Unix timestamp, session start timestamp. |
| waveCount | Integer  | Number of waves caught in the session, defaults to 0 |
| speedMax | Float  | Max speed attained in wave, will be null if no waves surfed. |
| distancePaddled | Float  | Total distance paddled. Can be null if the user hasn't paddled anywhere or the device doesn't provide this.|
| distanceWaves | Float  | Total distance surfed (sum of all wave distances). Null if no waves surfed.|
| calories | Integer  | Calories burned whilst surfing. Can be null if device doesn't provide this.|
| wavesPerHour | Integer  |Average waves caught per hour surfed. Can be null if no waves caught.|
| ratings | Integer | Automated rating for the Sessions time.  0 = VERY POOR, 1 = POOR, 2 = POOR TO FAIR, 3 = FAIR, 4 = FAIR TO GOOD |
| user | String | Unique user ID. |
| tide	| Float | Tide level in *feet*. |
| tide_normalized| Float | Normalized tide level. 0 = lowest tide possible, 1 = highest tide possible. |	
| next_tide| String | Next tide event, describing the tide level, type of extrema, and time of the extrema. |
| minBWH	| Float | Minimum breaking wave height in *feet*. |
| maxBWH| Float | Maximum breaking wave height in *feet*. |
| hour	| Integer |  1 = 1am, 2 = 2am, 13 = 1pm, etc. |
| month	| Integer |  1 = January, 2 = February, etc. |
| weekday| Integer |  0 = Monday, 1 = Tuesday, etc. |

---

### Event

All events are stored in `waves/{SESSIONID}.json`. Each file contains an array of Event objects.


#### Object Properties

| Parameter      | Type    | Description                                                                                      |
| -------------- | ------- | ------------------------------------------------------------------------------------------------ |
| startTimestamp | Integer      | UTC timestamp when the event started.                                                            |
| endTimestamp   | Integer       | UTC timestamp when the event ended.                                                              |
| type           | Enum         | Key to detonate the event type: `PADDLE` - Used when the user is paddling. `WAVE` - Used when the user is actively surfing a wave.                                                                 |
| outcome        | Enum        | Key to detonate the outcome for this event: `ATTEMPTED` - This wave event was attempted only. `CAUGHT` - This wave event was considered as having been caught. (default is `CAUGHT`)         |
| distance   | Float      | Distance of the wave event in *meters*.                                                              |
| speedMax       | Float        | Maximum speed the user traveled during this event in *meters per second*.                        |
| speedAverage   | Float         | Average speed the user traveled during this event in *meters per second*.                        |
| positions      | Array         | Array of Position objects that are included as part of this event.   |

---

### Position

Object used to describe a discrete position in a user's session.

#### Object Properties

| Parameter      | Type    | Description                                                                                      |
| -------------- | -------  | ------------------------------------------------------------------------------------------------ |
| timestamp      | Integer    | Unix timestamp in milliseconds when this position object occurred.                               |
| latitude       | Float        | Coorindate latitude, degrees, decimal format.                                                    |
| longitude      | Float       | Coorindate longitude, degrees, decimal format.                                                   |
| location |    Object    | Object containing the location info for the position. |
| location.type | String  | The type of spot location (Eg: POINT). |
| location.coordinates | Array | Coordinate array ([lon, lat]) |