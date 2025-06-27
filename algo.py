import json
import pandas as pd
import re
import openai
import os
from openai import OpenAI

import folium
from folium import Choropleth
from shapely.geometry import shape
import geopandas as gpd


# hardcoded OpenAI key - CHANGE FOR PUBLISHING
client = OpenAI(api_key="")
# load the GeoJSON file
with open("Composite.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# Extract properties into a df
features = geojson_data["features"]
all_properties = [feature["properties"] for feature in features]
df = pd.DataFrame(all_properties)


score_columns = ['ALC_Score', 'INTER_SCORE', 'DEST_SCORE', 'POP_SCORE',
                 'TRANSIT_SCORE', 'JOBS_SCORE', 'BIKE_SCORE', 'BLOS_SCORE']

# drop empty cols
for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=score_columns, how='all', inplace=True)

# LLM query interpreter
def interpret_query_with_llm(natural_query: str):
    prompt = f"""
    You are a helpful assistant that converts natural language queries about geographic region scores
    into structured filter instructions for a pandas DataFrame.

    Only respond with JSON containing score filters.

    Valid fields include: ALC_Score, INTER_SCORE, DEST_SCORE, POP_SCORE,
    TRANSIT_SCORE, JOBS_SCORE, BIKE_SCORE, BLOS_SCORE

    Valid comparisons: >, >=, <, <=, ==

    Example:
    Input: "show areas with high alcohol and low population"
    Output: {{
      "ALC_Score": ">= 4",
      "POP_SCORE": "<= 2"
    }}

    Now parse this:
    "{natural_query}"
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", #future to use gpt4
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    reply = response.choices[0].message.content.strip() #take first response, text content, and clean up whitespace
    try:
        return json.loads(reply)
    except:
        print("Could not parse LLM response:", reply)
        return {}

# Apply filters to df
def apply_filters(filter_dict, df):
    conditions = []
    for col, expr in filter_dict.items():
        if col in df.columns:
            #translation of expressions like >= 4
            match = re.match(r"(>=|<=|==|>|<)\s*(\d+(\.\d+)?)", expr.strip())
            if match:
                op, val, _ = match.groups()
                conditions.append(f"({col} {op} {val})")
    if not conditions:
        print("No valid filters interpreted.")
        return pd.DataFrame()
    query_string = " & ".join(conditions)
    return df.query(query_string)

def generate_summary_from_results(filtered_df):
    if filtered_df.empty:
        return "No results found, so nothing to summarize."

    sample = filtered_df.head(10)[score_columns + ['OBJECTID']].to_dict(orient='records')

    prompt = f"""
You are an urban planning analyst. Summarize the key patterns, risks, or insights from the following 10 geographic regions based on their scores.

Each region contains the following fields:
{', '.join(score_columns)}

Here are the regions:
{json.dumps(sample, indent=2)}

Respond with a short summary of what these regions represent. Be concise and insightful.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", #dont use gpt4
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


def visualize_results_on_map(filtered_df, original_geojson, save_path="map.html"):
    if filtered_df.empty:
        print("No data to visualize.")
        return

    # Build a GeoDataFrame from matching GeoJSON features - gpt
    matching_ids = set(filtered_df["OBJECTID"].astype(int))
    matching_features = [
        feature for feature in original_geojson["features"]
        if feature["properties"]["OBJECTID"] in matching_ids
    ]

    # create a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(matching_features)

    # Convert to WGS84 if needed - gpt
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)

    # pick a score column to color by (first available)
    color_by = next((col for col in score_columns if col in gdf.columns and gdf[col].notna().any()), None)

    # get center for the map
    centroid = gdf.geometry.union_all().centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12)

    # Add polygons
    folium.GeoJson(
        gdf,
        name="Filtered Regions",
        tooltip=folium.GeoJsonTooltip(fields=["OBJECTID"] + score_columns),
        style_function=lambda feature: {
            'fillColor': '#ff7800',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5,
        }
    ).add_to(m)

    # Save the map
    m.save(save_path)
    print(f"\n  Map saved to {save_path} — open in a browser to view.")


# ask user for a natural query
user_query = input("Ask a data question (e.g., 'regions with high job score and low transit'): ")
filters = interpret_query_with_llm(user_query)
result_df = apply_filters(filters, df)

# Show results, summary, then visualizer
print(f"\n Interpreted Filters: {filters}")
print("\n Matching Regions:\n", result_df[['OBJECTID'] + list(filters.keys())].head(10))

summary = generate_summary_from_results(result_df)
print("\n Summary of Results:\n", summary)

visualize_results_on_map(result_df, geojson_data)
