import duckdb
import pandas as pd
import plotly.graph_objects as go

CLEANED_PARQUET = "Medicare_Geo_Service_Cleaned.parquet"

con = duckdb.connect()

# 1. State Name to 2-Letter Postal Abbreviation mapping (Required for Plotly USA-states)
STATE_ABBR_MAP = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'Puerto Rico': 'PR', 'Virgin Islands': 'VI', 'Guam': 'GU'
}

# 2. Extract Top 5 Highest-Volume Procedures across the US
top_codes_df = con.execute(f"""
    SELECT 
        HCPCS_Code, 
        HCPCS_Desc,
        SUM(Total_Services) AS National_Services
    FROM read_parquet('{CLEANED_PARQUET}')
    WHERE Geo_Level = 'State'
    GROUP BY HCPCS_Code, HCPCS_Desc
    ORDER BY National_Services DESC
    LIMIT 50;
""").fetchdf()

top_codes = top_codes_df['HCPCS_Code'].tolist()

# 3. Pull State-level metrics for these top procedures
query = f"""
    SELECT 
        Geo_Desc AS State_Name,
        HCPCS_Code,
        HCPCS_Desc,
        Place_Of_Srvc_Desc,
        Total_Beneficiaries,
        Total_Services,
        Avg_Submitted_Charge,
        Avg_Medicare_Allowed_Amt,
        Avg_Medicare_Standardized_Amt,
        Billed_To_Allowed_Markup_Ratio
    FROM read_parquet('{CLEANED_PARQUET}')
    WHERE Geo_Level = 'State'
      AND HCPCS_Code IN ({', '.join([f"'{c}'" for c in top_codes])})
"""
df = con.execute(query).fetchdf()
df['State_Abbr'] = df['State_Name'].map(STATE_ABBR_MAP)
df = df.dropna(subset=['State_Abbr'])

# 4. Build Multi-Trace Plotly Choropleth with Dropdown Selector
fig = go.Figure()
buttons = []

for i, code in enumerate(top_codes):
    sub_df = df[df['HCPCS_Code'] == code]
    desc = top_codes_df.loc[top_codes_df['HCPCS_Code'] == code, 'HCPCS_Desc'].values[0]
    
    # Add a choropleth trace per HCPCS code
    fig.add_trace(
        go.Choropleth(
            locations=sub_df['State_Abbr'],
            z=sub_df['Avg_Medicare_Standardized_Amt'],
            locationmode='USA-states',
            colorscale='Blues',
            colorbar_title="Standardized ($)",
            name=f"{code}",
            visible=(i == 0),  # Only show the first procedure by default
            hovertemplate=(
                "<b>%{hovertext}</b><br><br>" +
                "HCPCS: " + code + "<br>" +
                "Avg Standardized Spend: $%{z:,.2f}<br>" +
                "Avg Billed Charge: $%{customdata[0]:,.2f}<br>" +
                "Markup Ratio: %{customdata[1]:.2f}x<br>" +
                "Total Services: %{customdata[2]:,}<br>" +
                "<extra></extra>"
            ),
            hovertext=sub_df['State_Name'],
            customdata=sub_df[['Avg_Submitted_Charge', 'Billed_To_Allowed_Markup_Ratio', 'Total_Services']]
        )
    )

    # Configure visibility array for the dropdown button
    visible_array = [False] * len(top_codes)
    visible_array[i] = True
    
    buttons.append(
        dict(
            label=f"{code} - {desc[:30]}...",
            method="update",
            args=[
                {"visible": visible_array},
                {"title": f"Regional Price Variation: {code} ({desc})"}
            ]
        )
    )

# 5. Add Dropdown UI & Map Layout
initial_desc = top_codes_df.iloc[0]['HCPCS_Desc']
fig.update_layout(
    title_text=f"Regional Price Variation: {top_codes[0]} ({initial_desc})",
    geo=dict(
        scope='usa',
        projection=go.layout.geo.Projection(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'
    ),
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )
    ],
    margin=dict(l=20, r=20, t=80, b=20)
)

# Render interactive map in browser
fig.show()

# Optional: Save as standalone interactive HTML
fig.write_html("regional_price_variation_map.html")
print("✅ Saved interactive choropleth map to regional_price_variation_map.html")