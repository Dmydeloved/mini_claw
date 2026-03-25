---
name: get_weather
description: Get current weather information for any city
trigger: weather|天气|forecast|温度|temperature
enabled: true
---

# Get Weather Skill

This skill teaches you how to fetch weather information for a given city.

## When to Use

Use this skill when the user asks about:
- Current weather conditions
- Temperature in a city
- Weather forecast
- Climate information

## Steps to Execute

1. **Extract the city name** from the user's message
   - Look for city names in the query
   - If no city is specified, ask the user which city they want

2. **Use the fetch_url tool** to get weather data
   - Use a weather API like wttr.in (no API key required)
   - Format: `https://wttr.in/{city}?format=j1`
   - Example: `fetch_url('https://wttr.in/Beijing?format=j1')`

3. **Parse the JSON response** using python_repl
   - Extract relevant information: temperature, conditions, humidity
   - Example code:
   ```python
   import json
   data = json.loads(response)
   current = data['current_condition'][0]
   temp_c = current['temp_C']
   weather_desc = current['weatherDesc'][0]['value']
   humidity = current['humidity']
   ```

4. **Format the response** in a user-friendly way
   - Present temperature in both Celsius and Fahrenheit
   - Include weather description
   - Add any relevant details (humidity, wind speed, etc.)

## Example Interaction

**User**: "What's the weather in Shanghai?"

**Your Actions**:
1. Call `fetch_url('https://wttr.in/Shanghai?format=j1')`
2. Parse the JSON response with python_repl
3. Respond: "The current weather in Shanghai is 22°C (72°F), partly cloudy with 65% humidity."

## Alternative APIs

If wttr.in is unavailable, you can also use:
- OpenWeatherMap API (requires API key)
- WeatherAPI.com (requires API key)

## Notes

- Always handle errors gracefully (city not found, API unavailable)
- Convert temperatures if user prefers different units
- Be concise but informative in your response
