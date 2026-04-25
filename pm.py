import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import os
import json

import pm4py
from colorize import colorize_net, get_colors, split_equal

import base64
from io import BytesIO

from pydantic import BaseModel
from typing import Any

class OutputModel(BaseModel):
    table: list[dict]
    image: str | None = None
    text: str | None = None
    plot: str | None = None
    big_plot: str | None = None
    process_model: str | None = None
    nodes: list[dict[str, Any]] | None = None
    edges: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None

class AnalysisFilterModel(BaseModel):
    metric: list[str]
    resource: list[str]
    role: list[str]
    activity: list[str]
    panel_id: str

pd.set_option('display.max_columns', None)

def plt_to_image(plt):
    # Write plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


#TODO add figsize as global var or as parameter
BLUE = "#2066a8"
GREEN = "#3a8158"

color_scale = [
   [0, "#2066a8"],
   [0.1, "#8ec1da"],
   [0.2, "#cde1ec"],
   [0.4, "#ededed"],
   [0.6, "#f6d6c2"],
   [0.8, "#d47264"],
   [1.0, "#ae282c"]
]

custom_palette_1 = [
    "#E68A75",
    "#75E68A",
    "#758AE6",
    "#E675A3",
    "#E6B875",
    "#75E6CC",
    "#B875E6",
    "#E6D475",
    "#8AE675",
    "#db6d53", 
    "#8A75E6",
    "#75E6C2",
    "#E675D4",
    "#E6A275",
    "#75B8E6",
    "#913ec9", 
    "#E6E675",
    "#75E6A2",
    "#E67575",
    "#75E675",
    "#478ebf", #replace
    "#A275E6",
    "#E6CC75",
    "#a7e8de"
]

custom_palette_2 = [
    "#758AE6",
    "#E675A3",
    "#E6B875",
    "#75E6CC",
    "#B875E6"
]

def get_node_hover_details(df):
    # Create a dictionary to store metrics for each activity
    activity_hover_details = {}
    
    # Get list of all unique activities
    activities = df["Activity"].unique()
    
    for selected_activity in activities:
        # Filter only the current activity
        activity_df = df[df["Activity"] == selected_activity].copy()
        
        if activity_df.empty:
            activity_hover_details[selected_activity] = {"error": f"No data found for activity: {selected_activity}"}
            continue
        
        # Count occurrences
        occurrence_count = len(activity_df)
        unique_cases = activity_df["Case ID"].nunique()
        unique_resources = activity_df["Resource"].nunique()
        
        # Get most common role
        responsible_role = activity_df["Role"].mode()[0] if not activity_df["Role"].empty else "Unknown"
        
        # Convert timestamps
        activity_df["Start Timestamp"] = pd.to_datetime(activity_df["Start Timestamp"])
        activity_df["Complete Timestamp"] = pd.to_datetime(activity_df["Complete Timestamp"])
        
        # Calculate durations
        activity_df["Duration"] = (activity_df["Complete Timestamp"] - activity_df["Start Timestamp"]).dt.total_seconds() / 60  # in minutes
        
        # Average case duration
        avg_duration = activity_df.groupby('Case ID')['Duration'].sum().mean()
        
        # Store metrics in dictionary
        activity_hover_details[selected_activity] = {
            "case_count": occurrence_count,
            "unique_cases": unique_cases,
            "unique_resources": unique_resources,
            "responsible_role": responsible_role,
            "average_duration": round(avg_duration, 2),
        }
    
    return activity_hover_details

def calculate_node_measures(df, selected_activity):
    unique_resources = df[df["Activity"] == selected_activity]["Resource"].nunique()
    responsible_role = df[df["Activity"] == selected_activity]["Role"].mode()[0]

    # Filter only the selected activity
    activity_df = df[df["Activity"] == selected_activity].copy()

    # Convert timestamps
    activity_df["Start Timestamp"] = pd.to_datetime(activity_df["Start Timestamp"])
    activity_df["Complete Timestamp"] = pd.to_datetime(activity_df["Complete Timestamp"])

    # Calculate durations
    activity_df["Duration"] = (activity_df["Complete Timestamp"] - activity_df["Start Timestamp"]).dt.total_seconds() / 60  # in minutes

    # Total work time per resource
    total_work_time = activity_df.groupby("Resource")["Duration"].sum().sum()  # all resources

    # Estimate available time: assume 7.7h/day, over the range of days in the log
    num_resources = activity_df["Resource"].nunique()
    date_range = df["Complete Timestamp"].max() - df["Start Timestamp"].min()
    num_days = date_range.days + 1
    available_minutes = num_resources * num_days * 7.7 * 60  # 8h per day in minutes

    capacity_utilization = (total_work_time / available_minutes) * 100

    # Total duration per resource per case
    resource_case_duration = activity_df.groupby(["Resource", "Case ID"])["Duration"].sum().reset_index()

    # Now average per resource
    acd_per_resource = resource_case_duration.groupby("Resource")["Duration"].mean()

    # Convert the Series to a DataFrame
    acd_per_resource_df = acd_per_resource.reset_index()
    acd_per_resource_df.columns = ['Resource', 'Average Case Duration [min]']

    fig = px.bar(acd_per_resource_df, y='Resource', x='Average Case Duration [min]', 
        title=f'Activity: {selected_activity}',
        labels={'Average Case Duration [min]': 'Average Case Duration [min]', 'Resource': 'Resource'},
        color_discrete_sequence=[BLUE])
    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  # Width in pixels
        height=400,  # Height in pixels
        title={
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    fig.update_traces(
        hovertemplate='Avg. Case Duration: %{customdata}', 
        customdata=round(acd_per_resource_df['Average Case Duration [min]'], 2))
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=round(acd_per_resource_df["Average Case Duration [min]"].max()/5), automargin=True)

    plot = fig.to_json()

    measures = pd.DataFrame([{
        "Activity": selected_activity,
        "Unique Resources": unique_resources,
        "Responsible Role": responsible_role,
        "Total Work Time (Minutes)": total_work_time,
        "Capacity Utilization (%)": f"{round(capacity_utilization*100, 2)} %",
        "Average Case Duration per Resource (Minutes)": round(acd_per_resource.mean(), 2),
    }])

    return OutputModel(table=measures.to_dict(orient="records"), plot=plot)

def filter_values_from_df(df):
    unique_values = {
        "Activity": df["Activity"].unique().tolist(),
        "Resource": df["Resource"].unique().tolist(),
        "Role": df["Role"].unique().tolist(),
        "Metric": []
    }
    return unique_values

def info_panel_file(df):
    file_path = os.path.join("data", "InfoPanel.json")

    # Ensure the file exists before trying to read it
    if not os.path.exists(file_path):
        return {"error": f"InfoPanel.json not found at {file_path}"}

    # Read the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    return data

def prettify_df(df):
    df_new = df.copy()
    for col,coltype in df_new.dtypes.to_dict().items():
        if coltype=='timedelta64[ns]':
            new_col = f"{col} (Minutes)"
            df_new[col] = (df_new[col].dt.total_seconds() / 60).round(2)
            #df_new.drop(col, axis=1, inplace=True)
            df_new.rename({col:new_col}, axis=1, inplace=True)

    for col,coltype in df_new.dtypes.to_dict().items():
        if coltype=='float64':
            df_new[col] = df_new[col].round(2)

    return df_new

def units_per_role(df):
    df = df.groupby('Role')['Resource'].nunique().reset_index()

    # Create a bar plot using matplotlib
    #plt.figure(figsize=(10, 6))
    #plt.bar(df['Role'], df['Resource'], color='skyblue')
    #plt.title('Number of Unique Resources per Role')
    #plt.xlabel('Role')
    #plt.ylabel('Number of Unique Resources')
    #plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

    # Create a bar plot using px
    fig = px.bar(df, x='Resource', y='Role',
        title='Number of Unique Resources per Role',
        labels={'Unique Resources': 'Unique Resources', 'Role': 'Role'},
        color_discrete_sequence=[BLUE],
        orientation="h")

    # Custom hover text
    fig.update_traces(hovertemplate='Resources: %{x}')
    fig.update_xaxes(title_text='Number of Resources')
    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )



    # Prepare return values
    table_data = df.to_dict(orient="records")
    #image_base64 = plt_to_image(plt)
    plot = fig.to_json()

    #return {
    #    "table": table_data,
    #    "image": image_base64,
    #    "text": None
    #}
    #return OutputModel(table=table_data, image=image_base64)
    return OutputModel(table=table_data, plot=plot)

def role_average_duration(df, normalize: bool = False):
    # Group by Role
    grouped_roles = df.groupby('Role')

    # Initialize lists to store data for each role
    unique_roles = []
    average_duration_per_case = []
    # Iterate through groups
    for role, role_df in grouped_roles:
        # Calculate and append the average duration per case for this role
        average_duration = role_df.groupby('Case ID')['Duration'].sum().mean()
        unique_roles.append(role)
        average_duration_per_case.append(average_duration)

    # Create a DataFrame for the result
    result_df = pd.DataFrame({
        "Role": unique_roles,
        "Average Case Duration": average_duration_per_case
    })

    # Highlight the role with the highest average case duration
    #result_df = result_df.style.apply(highlight_max, subset=['Average Case Duration'])

    # Perform min-max normalization on the 'Average Case Duration' column
    if normalize:
        result_df['Normalized Duration'] = (result_df['Average Case Duration'] - result_df['Average Case Duration'].min()) / (result_df['Average Case Duration'].max() - result_df['Average Case Duration'].min())

    #minutes = result_df.copy()
    # Convert 'Average Case Duration' from timedelta64[ns] to minutes
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)

    fig = px.bar(result_df, y='Role', x='Average Case Duration (Minutes)',
        title='Average Case Duration per Role (in Minutes)',
        labels={'Average Case Duration (Minutes)': 'Average Case Duration [min]', 'Role': 'Role'},
        orientation='h',
        color_discrete_sequence=[BLUE])

    # Custom hover text 
    fig.update_traces(hovertemplate='Average Case Duration: %{x:,.2f} minutes')

    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )

    plot = fig.to_json()
    process_model = activity_case_duration(df).process_model

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot, process_model=process_model)

def resource_average_duration(df):
    # TODO: highlight max value (in js)
    # Group by Resource
    grouped_resources = df.groupby('Resource')

    # Initialize lists to store data for each resource
    unique_resources = []
    average_duration_per_case = []

    # Iterate through groups
    for resource, resource_df in grouped_resources:
        # Calculate and append the average duration per case for this resource
        average_duration = resource_df.groupby('Case ID')['Duration'].sum().mean()
        unique_resources.append(resource)
        average_duration_per_case.append(average_duration)

    # Create a DataFrame for the result
    result_df = pd.DataFrame({
        "Resource": unique_resources,
        "Average Case Duration": average_duration_per_case
    })
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)

    return OutputModel(table=result_df.to_dict(orient="records"))

def resource_roles(df):
    # Initialize lists to store unique resources and their roles
    unique_resources = []
    roles_per_resource = []
    num_roles_per_resource = []

    # Group by Resource
    grouped_resources = df.groupby('Resource')

    # Iterate through groups
    for resource, resource_df in grouped_resources:
        # Append unique resource
        unique_resources.append(resource)
        # Append roles for the current resource as a list
        roles = resource_df['Role'].unique().tolist()
        roles_per_resource.append(roles)
        # Append the number of unique roles for the current resource
        num_roles_per_resource.append(len(roles))

    # Create a DataFrame from lists
    resource_role_df = pd.DataFrame({
        'Resource': unique_resources,
        'Number of Roles': num_roles_per_resource,
        'Roles': roles_per_resource
    })

    fig = px.bar(resource_role_df, y='Resource', x='Number of Roles', 
        title='Number of Roles per Resource',
        labels={'Number of Roles': 'Number of Roles', 'Resource': 'Resource'},
        color_discrete_sequence=[BLUE])
    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  # Width in pixels
        height=800,  # Height in pixels
        title={
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    fig.update_traces(hovertemplate='Roles: %{customdata}', customdata=resource_role_df['Roles'])# Update x-axis ticks to display only integer values
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)

    plot = fig.to_json()

    return OutputModel(table=resource_role_df.to_dict(orient="records"), plot=plot)


def _resource_role_matrix_colorscale(n_roles: int, palette: list[str]) -> list[list]:
    """Piecewise scale for z in 0..n_roles: 0 = empty cell, k>0 = role column color."""
    light_gray = "#ededed"
    if n_roles <= 0:
        return [[0, light_gray], [1, light_gray]]

    scale: list[list] = [[0, light_gray], [0.5 / n_roles, light_gray]]
    for k in range(1, n_roles + 1):
        color = palette[(k - 1) % len(palette)]
        t_mid = (k - 0.5) / n_roles
        t_end = k / n_roles
        scale.append([t_mid, color])
        scale.append([t_end, color])
    if scale[-1][0] < 1.0:
        scale.append([1.0, palette[(n_roles - 1) % len(palette)]])
    return scale


def resource_role_matrix(df):
    """
    Resource×Role assignment matrix (heatmap), Role–Activity edges, and summary metrics
    for organizational mining style views.
    """
    assignments = df[["Resource", "Role"]].dropna().drop_duplicates()
    roles_sorted = sorted(df["Role"].dropna().unique().tolist())
    n_roles = len(roles_sorted)

    all_resources = sorted(df["Resource"].dropna().unique().tolist())
    role_lists_by_resource = (
        assignments.groupby("Resource")["Role"]
        .agg(lambda s: sorted(s.unique().tolist()))
        if not assignments.empty
        else pd.Series(dtype=object)
    )

    roles_per_resource_rows = []
    for resource in all_resources:
        if resource in role_lists_by_resource.index:
            role_list = role_lists_by_resource[resource]
            if not isinstance(role_list, list):
                role_list = [role_list]
        else:
            role_list = []
        roles_per_resource_rows.append(
            {
                "Resource": resource,
                "Number of Roles": len(role_list),
                "Roles": role_list,
            }
        )

    meta = pd.DataFrame(roles_per_resource_rows)
    if meta.empty:
        meta = pd.DataFrame(columns=["Resource", "Number of Roles", "Roles"])
    else:
        meta["roles_key"] = meta["Roles"].apply(lambda r: tuple(r))
        meta = meta.sort_values(
            by=["Number of Roles", "roles_key", "Resource"],
            ascending=[False, True, True],
        )
    resources_sorted = meta["Resource"].tolist() if not meta.empty else []

    role_to_col = {role: j for j, role in enumerate(roles_sorted)}
    n_res = len(resources_sorted)
    z = [[0 for _ in range(n_roles)] for _ in range(n_res)]
    hover = [["" for _ in range(n_roles)] for _ in range(n_res)]

    res_index = {r: i for i, r in enumerate(resources_sorted)}
    for _, row in assignments.iterrows():
        resource = row["Resource"]
        role = row["Role"]
        if resource not in res_index or role not in role_to_col:
            continue
        i = res_index[resource]
        j = role_to_col[role]
        z[i][j] = j + 1
        hover[i][j] = f"Resource: {resource}<br>Role: {role}"

    resources_per_role_counts = (
        assignments.groupby("Role")["Resource"].nunique().astype(int).to_dict()
    )
    roles_per_resource_counts = (
        assignments.groupby("Resource")["Role"].nunique().astype(int).to_dict()
    )
    total_assignments = int(len(assignments))

    nodes: list[dict[str, Any]] = []
    for r in resources_sorted:
        nodes.append({"id": f"res:{r}", "label": r, "kind": "resource"})
    for role in roles_sorted:
        nodes.append({"id": f"role:{role}", "label": role, "kind": "role"})
    for activity in sorted(df["Activity"].dropna().unique().tolist()):
        nodes.append({"id": f"act:{activity}", "label": activity, "kind": "activity"})

    edges_rr: list[dict[str, Any]] = []
    for _, row in assignments.iterrows():
        resource = row["Resource"]
        role = row["Role"]
        edges_rr.append(
            {
                "source": f"res:{resource}",
                "target": f"role:{role}",
                "kind": "resource_role",
            }
        )

    role_activity = df[["Role", "Activity"]].drop_duplicates()
    edges_ra: list[dict[str, Any]] = []
    for _, row in role_activity.iterrows():
        role = row["Role"]
        activity = row["Activity"]
        edges_ra.append(
            {
                "source": f"role:{role}",
                "target": f"act:{activity}",
                "kind": "role_activity",
            }
        )

    edges = edges_rr + edges_ra

    metrics: dict[str, Any] = {
        "total_assignments": total_assignments,
        "resources_per_role": resources_per_role_counts,
        "roles_per_resource": roles_per_resource_counts,
    }

    row_height_px = 14
    base_h = n_res * row_height_px
    # Extra room for top titles, bottom count row, legend, and diagram controls (plotly default-like margins)
    plot_height = max(480, min(2600, base_h + 280))

    if n_roles == 0 or n_res == 0:
        fig = go.Figure()
        fig.update_layout(
            title={"text": "Resource × Role matrix", "x": 0.5, "xanchor": "center"},
            plot_bgcolor="white",
            annotations=[
                {
                    "text": "No Resource–Role assignments in the filtered log.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
    else:
        colorscale = _resource_role_matrix_colorscale(n_roles, custom_palette_1)
        palette = custom_palette_1
        per_role_count = [int(resources_per_role_counts.get(r, 0)) for r in roles_sorted]
        roles_count_per_res = [int(roles_per_resource_counts.get(r, 0)) for r in resources_sorted]
        max_role_count = max(roles_count_per_res) if roles_count_per_res else 1
        bar_axis_max = max(max_role_count * 1.12, 1.0)
        side_bar_gray = "#4a4a4a"
        # Thickness of each horizontal bar scales with the count (category-axis width, orientation=h)
        bar_thicknesses = [
            0.35 + 0.6 * (c / max_role_count) if max_role_count else 0.65 for c in roles_count_per_res
        ]
        # Top row: heatmap + per-resource count bars; bottom row: text counts under matrix columns, grand total
        fig = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.84, 0.12],
            column_widths=[0.70, 0.30],
            vertical_spacing=0.03,
            horizontal_spacing=0.002,
            subplot_titles=("", "Number of roles per resource", "", ""),
            specs=[[{}, {}], [{"type": "xy"}, {"type": "domain"}]],
        )

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=roles_sorted,
                y=resources_sorted,
                text=hover,
                hoverinfo="text",
                colorscale=colorscale,
                zmin=0,
                zmax=n_roles,
                showscale=False,
                showlegend=False,
                xgap=2,
                ygap=2,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=roles_count_per_res,
                y=resources_sorted,
                orientation="h",
                width=bar_thicknesses,
                marker=dict(color=side_bar_gray, line=dict(width=0)),
                text=[str(c) for c in roles_count_per_res],
                textposition="outside",
                textfont=dict(size=10, color=side_bar_gray),
                cliponaxis=False,
                showlegend=False,
                hovertemplate="Resource: %{y}<br>Number of roles: %{x}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        per_role_hover = [
            f"Role: {r}<br>Total unique resources: {c}" for r, c in zip(roles_sorted, per_role_count)
        ]
        fig.add_trace(
            go.Scatter(
                x=roles_sorted,
                y=[0.0] * n_roles,
                mode="text",
                text=[str(c) for c in per_role_count],
                textfont=dict(size=11, color="#333333"),
                textposition="middle center",
                showlegend=False,
                hovertext=per_role_hover,
                hoverinfo="text",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_assignments,
                number=dict(font=dict(size=28), valueformat="d"),
                title=dict(text="Total role<br>assignments", font=dict(size=12)),
            ),
            row=2,
            col=2,
        )

        for j, role in enumerate(roles_sorted):
            color = palette[j % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=role,
                    marker=dict(size=12, color=color, symbol="square"),
                    showlegend=True,
                    visible="legendonly",
                ),
                row=1,
                col=1,
            )

        # Shared column-1 x keeps counts aligned under each matrix column
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=roles_sorted,
            title_text="Role",
            showticklabels=True,
            tickangle=-40 if n_roles > 4 else 0,
            automargin=True,
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="Number of roles per resource",
            range=[0, bar_axis_max],
            showgrid=True,
            gridcolor="#eeeeee",
            row=1,
            col=2,
        )
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=roles_sorted,
            showticklabels=False,
            showgrid=False,
            title_text="Total unique resources (column total)",
            automargin=True,
            row=2,
            col=1,
        )
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)

        fig.update_yaxes(
            title_text="Resource",
            type="category",
            categoryorder="array",
            categoryarray=resources_sorted,
            autorange="reversed",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            type="category",
            categoryorder="array",
            categoryarray=resources_sorted,
            autorange="reversed",
            showticklabels=False,
            showgrid=True,
            gridcolor="#eeeeee",
            row=1,
            col=2,
        )
        # Numbers only, no y scale bar
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.4, 0.4],
            row=2,
            col=1,
        )

        # Margins: match typical bar/heatmap figures so toolbar controls do not sit on the plot
        fig.update_layout(
            title={
                "text": "Resource × Role matrix",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.9,
                "yanchor": "top",
            },
            height=plot_height,
            plot_bgcolor="white",
            margin=dict(t=100, b=100, l=80, r=80),
            legend=dict(
                title=dict(text="Role"),
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5,
            ),
        )

    plot = fig.to_json()
    table_records = meta.drop(columns=["roles_key"], errors="ignore").to_dict(orient="records")

    return OutputModel(
        table=table_records,
        plot=plot,
        nodes=nodes,
        edges=edges,
        metrics=metrics,
    )


#TODO: remove and replace by resource_within_role_normalization
def resource_role_average_duration(df, time_unit: str ='minutes', normalize: bool = True):
    # Group by Resource and Role
    grouped_resources_roles = df.groupby(['Resource', 'Role'])

    # Initialize lists to store data for each resource and role
    unique_resources = []
    roles = []
    average_duration_per_case = []

    # Iterate through groups
    for (resource, role), resource_role_df in grouped_resources_roles:
        # Append unique resource and role
        unique_resources.append(resource)
        roles.append(role)

        # Calculate and append the average duration per case for this resource and role
        average_duration = resource_role_df.groupby('Case ID')['Duration'].sum().mean()
        average_duration_per_case.append(average_duration)

    # Create a DataFrame for the result
    result_df = pd.DataFrame({
        "Resource": unique_resources,
        "Role": roles,
        "Average Case Duration": average_duration_per_case
    })

    # Perform min-max normalization on the 'Average Case Duration' column
    if normalize:
        result_df['Normalized Duration'] = (result_df['Average Case Duration'] - result_df['Average Case Duration'].min()) / (result_df['Average Case Duration'].max() - result_df['Average Case Duration'].min())

    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)

    return OutputModel(table=result_df.to_dict(orient="records"))

#TODO
# difference between resource_role_average_duration
# and resource_within_role_normalization

# TODO 2 plots for resource_within_role_normalization
def resource_within_role_normalization(df):
    # Group by Role
    grouped_by_role = df.groupby('Role')

    # list to store the result DataFrame for each role
    normalized_role_dfs = []

    for role, role_df in grouped_by_role:
        # Now group by Resource within this role
        grouped_resources = role_df.groupby('Resource')
        unique_resources = []
        average_duration_per_case = []
        for resource, resource_df in grouped_resources:
            # Calculate and append the average duration per case for this resource
            average_duration = resource_df.groupby('Case ID')['Duration'].sum().mean()
            unique_resources.append(resource)
            average_duration_per_case.append(average_duration)
        # Create a DataFrame for this role
        role_result_df = pd.DataFrame({
            "Role": [role] * len(unique_resources),
            "Resource": unique_resources,
            "Average Case Duration": average_duration_per_case
        })
        # Perform min-max normalization on the 'Average Case Duration' for this role
        min_duration = role_result_df['Average Case Duration'].min()
        max_duration = role_result_df['Average Case Duration'].max()
        role_result_df['Normalized Duration'] = (role_result_df['Average Case Duration'] - min_duration) / (max_duration - min_duration)

        # Append this role's DataFrame to the list
        normalized_role_dfs.append(role_result_df)

    # Concatenate all the role DataFrames
    normalized_df = pd.concat(normalized_role_dfs, ignore_index=True)

    fig = px.box(normalized_df, x='Normalized Duration', y='Role',
        title='Boxplot of Normalized Duration for Resources within Each Role',
        orientation='h',
        color_discrete_sequence=[BLUE])

    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )

    plot = fig.to_json()

    # Get unique roles & convert average case duration in minutes
    unique_roles = normalized_df['Role'].unique()

    # Create subplots: one row for each role
    fig = make_subplots(rows=len(unique_roles), cols=1, subplot_titles=[f'Role: {role}' for role in unique_roles])

    for i, role in enumerate(unique_roles, 1):
        # Filter the DataFrame for the current role
        role_df = normalized_df[normalized_df['Role'] == role]
        role_average_duration_minutes = (role_df['Average Case Duration'].dt.total_seconds()/ 60).round(2)
        fig.add_trace(
            go.Bar(
                y=role_df['Resource'], 
                x=role_average_duration_minutes,
                marker=dict(color=BLUE),
                hovertemplate='Average Case Duration: %{x} minutes<extra></extra>',
                orientation='h'
            ),
            row=i, 
            col=1
        )

    for i in range(1, len(unique_roles) + 1):
        fig.update_yaxes(title_text="Resource", row=i, col=1)
        fig.update_xaxes(title_text="Average Case Duration [min]", row=i, col=1)

    fig.update_layout(
        height=576 * len(unique_roles), 
        #width=1200, 
        title_text="Average Case Duration (in Minutes) per Role and Resource",
        plot_bgcolor='white',
        showlegend = False,
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )

    big_plot = fig.to_json()

    normalized_df['Average Case Duration (Minutes)'] = normalized_df['Average Case Duration'].dt.total_seconds() / 60
    normalized_df['Average Case Duration (Minutes)'] = normalized_df['Average Case Duration (Minutes)'].round(2)
    normalized_df.drop('Average Case Duration', axis=1, inplace=True)

    process_model = activity_case_duration(df).process_model

    return OutputModel(table=prettify_df(normalized_df).to_dict(orient="records"), plot=big_plot, big_plot=plot, process_model=process_model)


def roles_per_activity(df, count: bool = True):
    if count:
        result_df = df.groupby('Activity')['Role'].nunique().to_frame().reset_index()
    else:
        # TODO: output list in second column
        result_df = df.groupby(['Activity','Role']).size().reset_index().drop(0, axis=1)
    return OutputModel(table=result_df.to_dict(orient="records"))

def resources_per_activity(df, count: bool = True):
    if count:
        result_df = df.groupby("Activity")["Resource"].nunique().to_frame().reset_index()
    else:
        # TODO: output list in second column
        result_df = df.groupby(['Activity','Resource']).size().reset_index().drop(0, axis=1)

    result_df.rename({'Resource':'Number of Resources'}, axis=1, inplace=True)
    fig = px.bar(result_df, x='Number of Resources', y='Activity',
        title='Number of Unique Resources per Activity',
        color_discrete_sequence=[BLUE],
        orientation="h")

    # Custom hover text
    fig.update_traces(hovertemplate='Resources: %{x}')
    fig.update_xaxes(title_text='Number of Resources')
    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',  
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )
    plot = fig.to_json()
    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot)

def activities_per_role(df):
    result_df = df.groupby(['Role', 'Activity']).size().reset_index().drop(0, axis=1)

    # Sort roles by activity count
    activities_per_role_sorted = result_df.groupby('Role').size().reset_index(name='Activity Count').sort_values(by='Activity Count', ascending=False)
    # Create hover text for each role showing the activities performed by that role
    activities_per_role_sorted['Hover Text'] = activities_per_role_sorted.apply(lambda row: '<br>'.join(result_df[result_df['Role'] == row['Role']]['Activity']), axis=1)

    fig = px.bar(activities_per_role_sorted, x='Activity Count', y='Role',
        title='Number of Activities per Role',
        labels={'Activity Count': 'Number of Activities', 'Role': 'Role'},
        color_discrete_sequence=[BLUE]
        )
    fig.update_traces(hovertemplate='Activities: <br>%{customdata}', customdata=activities_per_role_sorted['Hover Text'])
    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    plot = fig.to_json()

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot)

def activities_per_role_new(df):
    unique_roles = []
    activities_per_role = []
    activitiy_count = []

    # Group by Resource
    grouped_roles = df.groupby('Role')

    # Iterate through groups
    for role, role_df in grouped_roles:
        unique_roles.append(role)
        # Append roles for the current resource as a list
        activities = ', '.join(role_df['Activity'].unique().tolist())
        activities_per_role.append(activities)
        # Append the number of roles for the current resource
        activitiy_count.append(role_df['Activity'].nunique())
        
    # Create a new df
    result_df = pd.DataFrame({
        'Role': unique_roles,
        'Number of Activities': activitiy_count,
        'Activities': activities_per_role
    })

    #return result_df
    plot_df = result_df.copy()
    hovertext = [activity.replace(', ', '<br>') for activity in plot_df['Activities']]

    fig = px.bar(plot_df, x='Number of Activities', y='Role',
        title='Number of Activities per Role',
        color_discrete_sequence=['#2066a8'])

    fig.update_traces(hovertemplate='Activities: <br>%{customdata}', customdata=hovertext)

    fig.update_layout(
        #width=1200, 
        height=576,  
        plot_bgcolor='white',  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()

    return OutputModel(table=result_df.to_dict(orient="records"), plot=plot)


def activity_average_duration_with_roles(df):
    # Group by Role and Activity
    grouped_role_activities = df.groupby(['Role', 'Activity'])

    # Initialize lists to store data
    unique_roles = []
    unique_activities = []
    average_duration_per_case = []

    # Iterate through each role and activity group
    for (role, activity), group_df in grouped_role_activities:
        # Calculate the sum of durations for each case, then find the mean
        average_duration = group_df.groupby('Case ID')['Duration'].sum().mean()
        # Append the results to the lists
        unique_roles.append(role)
        unique_activities.append(activity)
        average_duration_per_case.append(average_duration)

    # Create a DataFrame to display the results
    result_df = pd.DataFrame({
        "Role": unique_roles,
        "Activity": unique_activities,
        "Average Case Duration": average_duration_per_case
    })

    heatmap_roles = result_df.copy()
    heatmap_roles['Average Case Duration'] = (heatmap_roles['Average Case Duration'].dt.total_seconds()/60).round(2)
    heatmap_roles['Normalized Duration'] = heatmap_roles.groupby('Role')['Average Case Duration'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    all_roles = heatmap_roles['Role'].unique()
    all_activities = heatmap_roles['Activity'].unique()

    # Create pivot table
    table = heatmap_roles.pivot_table(index='Activity', columns='Role', values='Normalized Duration', aggfunc='mean', dropna=False)
    hover_table = heatmap_roles.pivot_table(index='Activity', columns='Role', values='Average Case Duration', aggfunc='mean', dropna=False)

    hover_text_hm = hover_table.applymap(lambda x: f'Average Case Duration: {x:.2f} minutes' if not np.isnan(x) else '')

    fig = go.Figure(data=go.Heatmap(
        z=table.values,
        x=table.columns,
        y=table.index,
        hoverinfo='text',
        text=hover_text_hm.values,
        colorscale=color_scale,
        colorbar=dict(
            title=dict(text='Average Case Duration (normalized)', side='right'),
            tickvals=[0, 1],
            ticktext=['Fastest<br>Resource', 'Slowest<br>Resource']
        ),
        showscale=True,
        xgap=2,
        ygap=2
    ))

    fig.update_layout(
        title={
            'text':'Normalized Average Case Duration per Activity and Role',
            'x':0.5,
            'xanchor': 'center'
        },
        xaxis=dict(title='Role'),
        yaxis=dict(title='Activity'),
        #width=1200,
        height=576,
        plot_bgcolor='white'
    )
    plot = fig.to_json()

    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)
    result_df['Normalized Duration'] = result_df.groupby('Role')['Average Case Duration (Minutes)'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot)

# TODO: activity_resource_comparison is the same just w/o norm?
def activity_resource_comparison(df, normalize: bool = False):
    # Group by Activity and then Resource
    grouped_activities = df.groupby(['Activity', 'Resource'])

    # Initialize a DataFrame to store the results
    result_list = []

    # Iterate through each activity and resource group
    for (activity, resource), group_df in grouped_activities:
        # Calculate the sum of durations for each case, then find the mean
        average_duration = group_df.groupby('Case ID')['Duration'].sum().mean()
        result_list.append({"Activity": activity, "Resource": resource, "Average Case Duration": average_duration})

    # Convert list to DataFrame
    result_df = pd.DataFrame(result_list)

    # Apply Min-Max Normalization within each activity
    if normalize:
        result_df['Normalized Duration'] = result_df.groupby('Activity')['Average Case Duration'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        #result_df['Normalized Duration'] = result_df['Normalized Duration'].astype('float64')
        #import numpy as np
        # TODO: optimize
        #result_df = result_df.replace({np.nan: None})

    if not normalize:
        unique_activities = result_df['Activity'].unique()

        fig = make_subplots(rows=len(unique_activities), cols=1, subplot_titles=[f'Activity: {activity}' for activity in unique_activities])

        for i, activity in enumerate(unique_activities, 1):
            activity_df = result_df[result_df['Activity'] == activity]
            activity_average_duration_minutes = (activity_df['Average Case Duration'].dt.total_seconds()/60).round(2)
            fig.add_trace(
                go.Bar(
                    y=activity_df['Resource'], 
                    x=activity_average_duration_minutes,
                    marker=dict(color=BLUE),
                    hovertemplate='Average Case Duration: %{x} minutes<extra></extra>',
                    orientation='h'
                ),
                row=i,
                col=1
            )
            max_duration = activity_average_duration_minutes.max() if not activity_average_duration_minutes.empty else 0
            fig.update_xaxes(range=[0, max_duration + 1], row=i, col=1)

        fig.update_layout(
            height=576 * len(unique_activities), 
            #width=1200,
            title_text="Average Case Duration (in Minutes) per Activity and Resource",
            showlegend=False,
            plot_bgcolor='white',
            title={
                'x':0.5,
                'xanchor': 'center'
            }
        )

        fig.update_xaxes(title_text="Average Case Duration [min]")
        fig.update_yaxes(title_text="Resource")

        big_plot=fig.to_json()

        result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
        result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
        result_df.drop('Average Case Duration', axis=1, inplace=True)

        result_df = result_df.replace({np.nan: None})

        return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), big_plot=big_plot)

    if normalize:
        normalized_data = result_df.copy()

        # Convert 'Average Case Duration' from Timedelta to minutes
        #normalized_data['Average Case Duration (Minutes)'] = (normalized_data['Average Case Duration'].dt.total_seconds() / 60).round(2)
        #normalized_data.drop('Average Case Duration', axis=1, inplace=True)
        normalized_data['Average Case Duration'] = (normalized_data['Average Case Duration'].dt.total_seconds() / 60).round(2)
        normalized_data.rename({'Average Case Duration':'Average Case Duration (Minutes)'}, axis=1, inplace=True)

        # Prepare the data for the heatmap
        pivot_table = normalized_data.pivot(index='Activity', columns='Resource', values='Normalized Duration')

        # Define the hover text to show both the average duration in minutes and the normalized value
        hover_text = normalized_data.pivot(index='Activity', columns='Resource', values='Average Case Duration (Minutes)')
        hover_text = hover_text.applymap(lambda x: f'Average Case Duration: {x:.2f} minutes' if pd.notnull(x) else '')

        # Now create the heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            hoverinfo='text',
            text=hover_text.values,
            colorscale=color_scale,
            colorbar=dict(
                title=dict(text='Average Case Duration (normalized)', side='right'),
                tickvals=[0, 1],
                ticktext=['Fastest<br>Resource', 'Slowest<br>Resource']),
            showscale=True,
            xgap=2,
            ygap=2
        ))

        # Update the layout
        fig.update_layout(
            title=dict(text='Normalized Average Case Duration per Activity and Resource',x=0.5, xanchor='center'),
            xaxis=dict(title='Resource'),
            yaxis=dict(title='Activity'),
            #width=1200,
            height=800,
            plot_bgcolor='white'
        )

        plot = fig.to_json()

        unique_activities_norm = normalized_data['Activity'].unique()
        fig = go.Figure()

        # Add a box trace for each activity
        for activity in unique_activities_norm:
            activity_df = normalized_data[normalized_data['Activity'] == activity]
            fig.add_trace(go.Box(
                x=activity_df['Normalized Duration'],
                name=activity,
                hoverinfo='none',  #disables hover information
                marker_color=BLUE
            ))

        # Update the layout
        fig.update_layout(
            xaxis_title='Normalized Duration',
            yaxis_title='Activity',
            #width=1200,  # Width in pixels
            height=576,  # Height in pixels
            plot_bgcolor='white',
            showlegend=False,
            title={
                'text':'Boxplot of Normalized Duration for Resources within Each Activity',
                'x':0.5,
                'xanchor': 'center'
            }
        )
        big_plot = fig.to_json()
        return OutputModel(table=prettify_df(normalized_data).replace({np.nan: None}).to_dict(orient="records"), plot=plot, big_plot=big_plot)

def slowest_resource_per_activity(df):
    # Group by Activity and then Resource
    grouped_activities = df.groupby(['Activity', 'Resource'])

    # Initialize a dictionary to store the slowest resource for each activity
    slowest_resources = {}

    # Iterate through each activity and resource group
    for (activity, resource), group_df in grouped_activities:
        # Calculate the sum of durations for each case, then find the mean
        average_duration = group_df.groupby('Case ID')['Duration'].sum().mean()

        # Update the entry for the activity with the slowest resource
        if activity not in slowest_resources or slowest_resources[activity][1] < average_duration:
            slowest_resources[activity] = (resource, average_duration)

    # Convert the dictionary to a DataFrame for better visualization
    result_df = pd.DataFrame(list(slowest_resources.items()), columns=['Activity', 'Slowest Resource Info'])
    result_df[['Slowest Resource', 'Average Case Duration']] = pd.DataFrame(result_df['Slowest Resource Info'].tolist(), index=result_df.index)
    result_df.drop('Slowest Resource Info', axis=1, inplace=True)

    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration'].dt.total_seconds() / 60
    result_df['Average Case Duration (Minutes)'] = result_df['Average Case Duration (Minutes)'].round(2)
    result_df.drop('Average Case Duration', axis=1, inplace=True)


    slowest = result_df.copy()
    #slowest['Average Case Duration'] = (slowest['Average Case Duration'].dt.total_seconds() / 60).round(2)

    fig = px.bar(slowest, 
        y='Activity', 
        x='Average Case Duration (Minutes)', 
        title='Slowest Resource per Activity',
        labels={'Average Case Duration': 'Average Case Duration [min]'},
        orientation='h',
        color_discrete_sequence=[BLUE]
        )

    hover_template = "Slowest Resource: %{customdata}<br> Average Case Duration: %{x} minutes"

    fig.update_traces(hovertemplate=hover_template,customdata=slowest['Slowest Resource'])

    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        title={
            'x':0.5,
            'xanchor': 'center'
        }
    )
    plot = fig.to_json()
    #return result_df
    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot)

#### Extension ####
def get_timeframe(df):
    return (df['Start Timestamp'].min(), df['Complete Timestamp'].max())

def calculate_working_days(start_date, end_date):
    # Calculate the number of working days (excluding weekends) between start and end date
    all_days = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' represents business days (Mon-Fri)
    return len(all_days)

# Advanced capacity calculation
# helper functions
# how many hours has a resource worked in the entire event log
def worked_hours_per_resource(df):
    result_df = df.groupby('Resource')['Duration'].sum().reset_index()

    result_df = result_df.rename(columns={'Duration': 'Total Time Worked'})

    #Convert to hours
    result_df['Total Time Worked (hr)'] = (result_df['Total Time Worked'].dt.total_seconds() / 3600).round(2)

    return result_df

# how many hours has the resource worked each day
# using first activity start as login
# and last activity end as logoff
def calculate_working_hours_per_day(df):
    # Add a date column
    df_with_date = df
    df_with_date['Date'] = df_with_date['Start Timestamp'].dt.date

    # Group by Resource and Date to calculate working hours
    working_hours_per_day = df_with_date.groupby(['Resource', 'Date']).agg(Login=('Start Timestamp', 'min'), Logout=('Complete Timestamp', 'max')).reset_index()

    # Calculate the working hours by subtracting start from end
    working_hours_per_day['Available Working Hours per Day'] = ((working_hours_per_day['Logout'] - working_hours_per_day['Login']).dt.total_seconds() / 3600).round(2)

    return working_hours_per_day

# how many hours a resource was available for work in the entire event log
def calculate_total_working_hours_per_resource(df):

    # Group by Resource, sum working hours per day to get the total working hours per resource within the process cycle
    result_df = calculate_working_hours_per_day(df).groupby('Resource')['Available Working Hours per Day'].sum().reset_index()

    result_df = result_df.rename(columns={'Available Working Hours per Day': 'Total Available Working Hours'})

    return result_df


# Resource capacity
def total_duration_per_activity_and_resource(df):
    # Group by Activity and Resource and sum durations to get the total time spent on each activity by each resource
    result_df = df.groupby(['Activity', 'Resource'])['Duration'].sum().reset_index()

    # Rename columns for better readability
    result_df=result_df.rename(columns={'Duration': 'Total Time Spent'})

    return result_df

def total_duration_per_resource_and_activity(df):
    # Group by Resource and Activity, sum durations to get the total time spent on each activity by each resource
    result_df = df.groupby(['Resource', 'Activity'])['Duration'].sum().reset_index()

    # Rename columns for better readability
    result_df=result_df.rename(columns={'Duration': 'Total Time Spent'})

    # Add a new column for the overall time spent in minutes
    result_df['Total Time Spent (min)'] = (result_df['Total Time Spent'].dt.total_seconds() / 60).round(2)  # Convert to minutes

    # Calculate the total time spent per resource on all activities
    total_time_per_resource = result_df.groupby('Resource')['Total Time Spent'].sum().reset_index()
    total_time_per_resource = total_time_per_resource.rename(columns={'Total Time Spent': 'Overall Time Spent'})

    # Merge the total time spent per resource into the result dataframe
    result_df = result_df.merge(total_time_per_resource, on='Resource')

    # Calculate the percentage of time spent on each activity
    result_df['Percentage Time Spent (%)'] = (result_df['Total Time Spent'] / result_df['Overall Time Spent'] * 100).round(2)

    # Drop the 'Overall Time Spent' column to hide it
    result_df = result_df.drop(columns=['Overall Time Spent'])

    #return result_df
    resource_time_distribution_plot = result_df.copy()
    unique_activities = resource_time_distribution_plot['Activity'].unique()
    color_palette = custom_palette_1 #sns.color_palette("husl", len(unique_activities)).as_hex()

    # dictionary for dfg color coding
    activity_color_mapping = {activity: color_palette[i] for i, activity in enumerate(unique_activities)}

    fig = px.bar(
        resource_time_distribution_plot,
        x='Percentage Time Spent (%)',
        y='Resource',
        color='Activity',
        orientation='h', 
        title='Percentage of Time Spent on Each Activity by Resource',
        labels={'Percentage Time Spent (%)': 'Percentage of Total Time Spent'},
        barmode='stack',
        custom_data=['Activity', 'Percentage Time Spent (%)'], 
        color_discrete_sequence=color_palette 
    )

    # Custom hover text
    fig.update_traces(
        hovertemplate='Activity: %{customdata[0]}<br>Relative duration: %{customdata[1]}%' +
            '<extra></extra>'  # Remove extra information
    )

    fig.update_layout(
        plot_bgcolor='white',
        #showlegend=False,
        #width=1200,  
        height=800,  
        xaxis=dict(dtick=10,title=dict(standoff=35)),
        #yaxis=dict(pad=10),
        bargap=0.3,
        margin = {'pad': 15},
        title={
            'y':0.92,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            y=-0.2,
            xanchor='center',  
            x=0.45  
        )
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')

    #colors = get_colors(dict(zip(result_df['Activity'], result_df['Normalized Average Case Duration'])), ctype="sat", base_color="#FFFFFF")
    process_model = colorize_net(heu_net, activity_color_mapping)

    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot, process_model=process_model)


def total_duration_per_resource(df):
    # Group by Resource and sum all durations to get the overall time spent by each resource
    result_df = df.groupby('Resource')['Duration'].sum().reset_index()

    # Rename the columns for better readability
    result_df = result_df.rename(columns={'Duration': 'Overall Time Spent'})

    # Add a new column for the overall time spent in hours, assuming 'Total Time Spent' is in minutes
    result_df['Overall Time Spent (Hours)'] = (result_df['Overall Time Spent'].dt.total_seconds()/3600).round(2)  # Convert minutes to hours

    return result_df

def capacity_utilization_resource(df, work_hours_per_day=7.7, activity=""):

    if activity != "":
        df = df[df['Activity'] == activity]

    start_date, end_date = get_timeframe(df)
    # get the overall time spent for each resource
    result_df = total_duration_per_resource(df)

    # Calculate the total available working hours during the event log period
    working_days = calculate_working_days(start_date, end_date)
    available_working_hours = working_days * work_hours_per_day

    # Calculate the capacity utilization for each resource
    result_df['Capacity Utilization (%)'] = (result_df['Overall Time Spent (Hours)'] / available_working_hours * 100).round(2)

    #return result_df
    capacity_utilization_resource_plot = result_df.copy()

    fig = px.bar(capacity_utilization_resource_plot, x='Capacity Utilization (%)', y='Resource',
        title='Capacity Utilization per Resource',
        labels={'Resource': 'Resource', 'Capacity Utilization (%)': 'Capacity Utilization (%)'},
        color_discrete_sequence=['#2066a8'],
        orientation="h")

    fig.update_traces(hovertemplate='Capacity Utilization: %{x}%')

    fig.add_shape(
        type="line",
        x0=100, x1=100,  # Vertical line at 100% on the x-axis
        y0=0, y1=1,      # Full height of the plot
        xref='x', yref='paper',  # x in data coordinates, y in paper (0 to 1) coordinates
        line=dict(color="red", width=2) #, dash="dash")  # Customizing the line style
    )

    fig.update_layout(
       # width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        xaxis=dict(dtick=10),  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()
    process_model = capacity_utilization_activity(df, work_hours_per_day).process_model
    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot, process_model=process_model)

def capacity_utilization_resource_new(df):
    # get the overall time spent for each resource
    worked_hours = worked_hours_per_resource(df)

    working_hours = calculate_total_working_hours_per_resource(df)

    # Merge the two DataFrames on the Resource column
    result_df = pd.merge(worked_hours, working_hours, on='Resource', how='inner')

    result_df.drop(columns=['Total Time Worked'], inplace=True)

    # Calculate the capacity utilization for each resource
    result_df['Capacity Utilization (%)'] = (result_df['Total Time Worked (hr)'] / result_df['Total Available Working Hours'] * 100).round(2)

    #return result_df
    capacity_utilization_resource_plot_new = result_df.copy()
    fig = px.bar(capacity_utilization_resource_plot_new, x='Capacity Utilization (%)', y='Resource',
        title='Capacity Utilization per Resource',
        #labels={'Resource': 'Resource', 'Capacity Utilization (%)': 'Capacity Utilization (%)'},
        color_discrete_sequence=['#2066a8'],
        orientation="h",
        category_orders={"Resource": capacity_utilization_resource_plot_new['Resource'].tolist()})

    fig.update_traces(hovertemplate='Capacity Utilization: %{x}%')

    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        xaxis=dict(dtick=10),  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()
    process_model = capacity_utilization_activity_new(df).process_model
    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot, process_model=process_model)

# Percentage of time spent by resource in a role
def workload_distribution_per_resource(df):
    # get time spent in each role for every resource
    total_time_role = df.groupby(['Resource', 'Role'])['Duration'].sum().reset_index()

    total_time_role = total_time_role.rename(columns={'Duration': 'Total Time Worked in Role (hr)'})

    # convert to hours
    total_time_role['Total Time Worked in Role (hr)'] = (total_time_role['Total Time Worked in Role (hr)'].dt.total_seconds() / 3600).round(2)

    #get the overall time spent for each resource using previously defined function
    total_time_resource = worked_hours_per_resource(df)

    # Merge dfs
    result_df = pd.merge(total_time_role, total_time_resource, on='Resource')
    
    # calculate relative time
    result_df['Percentage (%)'] = ((result_df['Total Time Worked in Role (hr)'] / result_df['Total Time Worked (hr)']) * 100).round(2)
    
    # drop irrelevant columns
    result_df.drop(columns=['Total Time Worked', 'Total Time Worked (hr)'], inplace=True)
    
    #return result_df
    workload_distribution_per_resource_plot = result_df.copy()
    unique_roles = workload_distribution_per_resource_plot['Role'].unique()
    color_palette = custom_palette_2 #sns.color_palette("husl", len(unique_roles)).as_hex()

    role_color_mapping = {role: color_palette[i] for i, role in enumerate(unique_roles)}

    fig = px.bar(workload_distribution_per_resource_plot, x='Percentage (%)', y='Resource',
        color='Role',
        title='Time Spent in Each Role by Resource',
        labels={'Percentage (%)': 'Percentage of Time Spent [%]'},
        orientation="h",
        category_orders={'Resource': workload_distribution_per_resource_plot['Resource'].tolist()},
        custom_data=['Role', 'Percentage (%)'],
        barmode='stack',
        color_discrete_sequence=color_palette)

    fig.update_traces(hovertemplate='Role: %{customdata[0]}<br>Total Time Spent: %{customdata[1]}%' + '<extra></extra>')

    fig.update_layout(
        #width=1200,  # Width in pixels
        height=800,  # Height in pixels
        plot_bgcolor='white',
        margin = {'pad': 15},
        xaxis=dict(dtick=10),  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            y=-0.2,
            xanchor='center',  
            x=0.45  
        )
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')

    activity_role_df = pd.DataFrame(activities_per_role(df).table)
    unique_activities = activity_role_df['Activity'].unique()
    activity_color_mapping = {activity: split_equal([role_color_mapping[role[1]] for role in activity_role_df.loc[activity_role_df['Activity']==activity, 'Role'].items()]) for activity in unique_activities}
    print(activity_color_mapping)
    #colors = get_colors(dict(zip(result_df['Activity'], result_df['Normalized Average Case Duration'])), ctype="sat", base_color="#FFFFFF")
    process_model = colorize_net(heu_net, activity_color_mapping)

    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot, process_model=process_model)


# Role capacity
# Naive
def total_duration_per_role(df):
    grouped_roles = df.groupby('Role')

    roles = []
    nr_resources = []
    overall_time_spent = []

    for role, role_df in grouped_roles:
        roles.append(role)
        nr_resources.append(role_df['Resource'].nunique())
        overall_time_spent.append(role_df['Duration'].sum())

    # Create a new df
    result_df = pd.DataFrame({
        'Role': roles,
        'Number of Resources': nr_resources,
        'Overall Time Spent': overall_time_spent
    })

    result_df['Overall Time Spent (Hours)'] = (result_df['Overall Time Spent'].dt.total_seconds()/3600).round(2)  # Convert minutes to hours

    return result_df

def total_duration_per_role_and_activity(df):
    
    # Group by Role and Activity, sum durations to get the total time spent on each activity by each role
    result_df = df.groupby(['Role', 'Activity'])['Duration'].sum().reset_index()

    # Rename columns for better readability
    result_df = result_df.rename(columns={'Duration': 'Total Time Spent'})

    # Add a new column for the overall time spent in minutes
    result_df['Total Time Spent (min)'] = (result_df['Total Time Spent'].dt.total_seconds() / 60).round(2)  # Convert to minutes

    # Calculate the total time spent per resource on all activities
    total_time_per_role = result_df.groupby('Role')['Total Time Spent'].sum().reset_index()
    total_time_per_role = total_time_per_role.rename(columns={'Total Time Spent': 'Overall Time Spent'})

    # Merge the total time spent per resource into the result dataframe
    result_df = result_df.merge(total_time_per_role, on='Role')

    # Calculate the percentage of time spent on each activity
    result_df['Percentage Time Spent (%)'] = (result_df['Total Time Spent'] / result_df['Overall Time Spent'] * 100).round(2)

    # Drop the 'Overall Time Spent' column to hide it
    result_df = result_df.drop(columns=['Overall Time Spent'])

    #return result_df
    role_time_distribution_plot = result_df.copy()
    unique_activities = role_time_distribution_plot['Activity'].unique()
    color_palette = custom_palette_1 #sns.color_palette("husl", len(unique_activities)).as_hex()

    # dictionary for dfg color coding
    activity_color_mapping = {activity: color_palette[i] for i, activity in enumerate(unique_activities)}

    fig = px.bar(
        role_time_distribution_plot,
        x='Percentage Time Spent (%)',
        y='Role',
        color='Activity',
        orientation='h', 
        title='Percentage of Time Spent on Each Activity by Role',
        labels={'Percentage Time Spent (%)': 'Percentage of Total Time Spent'},
        barmode='stack',
        custom_data=['Activity', 'Percentage Time Spent (%)'], 
        color_discrete_sequence=color_palette 
    )

    # Custom hover text
    fig.update_traces(
        hovertemplate='Activity: %{customdata[0]}<br>Relative duration: %{customdata[1]}%' +
            '<extra></extra>'  # Remove extra information
    )

    fig.update_layout(
        plot_bgcolor='white',
        #showlegend=False,
        #width=1200,  
        height=800,  
        xaxis=dict(dtick=10,title=dict(standoff=35)),
        #yaxis=dict(pad=10),
        bargap=0.5,
        margin = {'pad': 15},
        title={
            'y':0.92,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            y=-0.2,
            xanchor='center',  
            x=0.45  
        )
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')

    #colors = get_colors(dict(zip(result_df['Activity'], result_df['Normalized Average Case Duration'])), ctype="sat", base_color="#FFFFFF")
    process_model = colorize_net(heu_net, activity_color_mapping)

    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot, process_model=process_model)

def capacity_utilization_role(df, work_hours_per_day=7.7):
    start_date, end_date = get_timeframe(df)

    # get the overall time spent for each role
    result_df = total_duration_per_role(df)

    # Calculate the total available working hours during the event log period
    working_days = calculate_working_days(start_date, end_date)
    available_working_hours = working_days * work_hours_per_day

    # Calculate the capacity utilization for each role
    result_df['Capacity Utilization (%)'] = (result_df['Overall Time Spent (Hours)'] / (result_df['Number of Resources'] * available_working_hours) * 100).round(2)

    #return result_df
    capacity_utilization_role_plot = result_df.copy()

    fig = px.bar(capacity_utilization_role_plot, x='Capacity Utilization (%)', y='Role',
        title='Capacity Utilization per Role',
        labels={'Role': 'Role', 'Capacity Utilization (%)': 'Capacity Utilization (%)'},
        color_discrete_sequence=['#2066a8'],
        orientation="h")

    fig.update_traces(hovertemplate='Capacity Utilization: %{x}%')

    fig.add_shape(
        type="line",
        x0=100, x1=100,  # Vertical line at 100% on the x-axis
        y0=0, y1=1,      # Full height of the plot
        xref='x', yref='paper',  # x in data coordinates, y in paper (0 to 1) coordinates
        line=dict(color="red", width=2) #, dash="dash")  # Customizing the line style
    )

    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        xaxis=dict(dtick=10),  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()
    process_model = capacity_utilization_activity(df, work_hours_per_day).process_model
    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot, process_model=process_model)

# Advanced
def capacity_resource_role(df):
    workload = pd.DataFrame(workload_distribution_per_resource(df).table)
    availability = calculate_total_working_hours_per_resource(df)

    result_df = pd.merge(workload, availability, on='Resource', how='inner')

    result_df['Total Available Working Hours for this Role']= ((result_df['Percentage (%)']/100)*result_df['Total Available Working Hours']).round(2)

     # drop irrelevant columns
    result_df.drop(columns=['Total Available Working Hours'], inplace=True)

    return result_df

def capacity_utilization_role_new(df):
    role_duration = df.groupby('Role')['Duration'].sum().reset_index()

    role_duration['Duration'] = (role_duration['Duration'].dt.total_seconds() / 3600).round(2)

    role_duration = role_duration.rename(columns={'Duration': 'Total Time Worked (hr)'})

    role_capacity = capacity_resource_role(df).groupby('Role')['Total Available Working Hours for this Role'].sum().reset_index()

    result_df = pd.merge(role_duration, role_capacity, on='Role', how='inner')

    # Calculate the capacity utilization for each role
    result_df['Capacity Utilization (%)'] = (result_df['Total Time Worked (hr)'] / result_df['Total Available Working Hours for this Role'] * 100).round(2)

    #return result_df
    capacity_utilization_role_plot = result_df.copy()
    fig = px.bar(capacity_utilization_role_plot, x='Capacity Utilization (%)', y='Role',
        title='Capacity Utilization per Role',
        labels={'Role': 'Role', 'Capacity Utilization (%)': 'Capacity Utilization (%)'},
        color_discrete_sequence=['#2066a8'],
        orientation="h")

    fig.update_traces(hovertemplate='Capacity Utilization: %{x}%')

    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        xaxis=dict(dtick=10),  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()
    process_model = capacity_utilization_activity_new(df).process_model
    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot, process_model=process_model)

# Activity capacity
# Naive
def total_duration_per_activity(df):
    grouped_activities = df.groupby('Activity')

    activities = []
    nr_resources = []
    overall_time_spent = []

    for activity, activity_df in grouped_activities:
        activities.append(activity)
        nr_resources.append(activity_df['Resource'].nunique())
        overall_time_spent.append(activity_df['Duration'].sum())

    # Create a new df
    result_df = pd.DataFrame({
        'Activity': activities,
        'Number of Resources': nr_resources,
        'Overall Time Spent': overall_time_spent
    })

    result_df['Overall Time Spent (Hours)'] = (result_df['Overall Time Spent'].dt.total_seconds()/3600).round(2)  # Convert minutes to hours


    return result_df

def capacity_utilization_activity(df, work_hours_per_day=7.7):
    start_date, end_date = get_timeframe(df)

    # get the overall time spent for each activity
    result_df = total_duration_per_activity(df)

    # Calculate the total available working hours during the event log period
    working_days = calculate_working_days(start_date, end_date)
    available_working_hours = working_days * work_hours_per_day

    # Calculate the capacity utilization for each activity
    result_df['Capacity Utilization (%)'] = (result_df['Overall Time Spent (Hours)'] / (result_df['Number of Resources'] * available_working_hours) * 100).round(2)

    #return result_df
    capacity_utilization_activity_plot = result_df.copy()
    fig = px.bar(capacity_utilization_activity_plot, x='Capacity Utilization (%)', y='Activity',
             title='Capacity Utilization per Activity',
             labels={'Role': 'Role', 'Capacity Utilization (%)': 'Capacity Utilization (%)'},
             color_discrete_sequence=['#2066a8'],
             orientation="h")

    fig.update_traces(hovertemplate='Capacity Utilization: %{x}%')

    fig.add_shape(
        type="line",
        x0=100, x1=100,  # Vertical line at 100% on the x-axis
        y0=0, y1=1,      # Full height of the plot
        xref='x', yref='paper',  # x in data coordinates, y in paper (0 to 1) coordinates
        line=dict(color="red", width=2) #, dash="dash")  # Customizing the line style
    )

    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        xaxis=dict(dtick=10),  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')
    #graph = pm4py.visualization.heuristics_net.visualizer.get_graph(heu_net)
    #graph = assign_colors(graph, colors = {k:f"#FF0000;{random():.2f}:#FFFFFF" for k in heu_net.activities})

    colors = get_colors(dict(zip(result_df['Activity'], result_df['Capacity Utilization (%)'] / 100)))
    #print(colors)
    #png_image = graph.create_png()
    #process_model = base64.b64encode(png_image).decode('utf-8')
    process_model = colorize_net(heu_net, colors)

    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot, process_model=process_model)


def activity_case_duration(df):
    # Group by Activity
    grouped_activities = df.groupby('Activity')

    unique_activities = []
    average_duration_per_case = []
    median_duration_per_case = []

    # Iterate through groups
    for activity, activity_df in grouped_activities:
        # Calculate total duration per case
        case_durations = activity_df.groupby('Case ID')['Duration'].sum()

        # Calculate and append avg/median case duration
        average_duration = case_durations.mean()
        #median_duration = case_durations.median()

        unique_activities.append(activity)
        average_duration_per_case.append(average_duration)
        #median_duration_per_case.append(median_duration)

    # Create resulting df
    result_df = pd.DataFrame({
        "Activity": unique_activities,
        "Average Case Duration": average_duration_per_case #,
        #"Median Case Duration": median_duration_per_case
    })
    
    ## Perform min-max normalization using lambda functions on both Average and Median Case Duration
    result_df['Normalized Average Case Duration'] = result_df['Average Case Duration'].apply(
        lambda x: (x - result_df['Average Case Duration'].min()) / (result_df['Average Case Duration'].max() - result_df['Average Case Duration'].min()))
        
    #result_df['Normalized Median Case Duration'] = result_df['Median Case Duration'].apply(
    #    lambda x: (x - result_df['Median Case Duration'].min()) / (result_df['Median Case Duration'].max() - result_df['Median Case Duration'].min()))

    #return result_df

    acd_activity = result_df.copy()

    # Convert ACD from timedelta64[ns] to minutes
    acd_activity['Average Case Duration (Minutes)'] = (acd_activity['Average Case Duration'].dt.total_seconds()/ 60).round(2)

    fig = px.bar(acd_activity, y='Activity', x='Average Case Duration (Minutes)',
        title='Average Case Duration per Activity (in Minutes)',
        #labels={'Average Case Duration (Minutes)': 'Average Case Duration [min]', 'Activity': 'Activity'},
        orientation='h',
        color_discrete_sequence=['#2066a8'])

    # Custom hover text 
    fig.update_traces(hovertemplate='Average Case Duration: %{x:,.2f} minutes')

    fig.update_layout(
        plot_bgcolor='white',
        #width=1200,  
        height=576,  
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')
    #graph = pm4py.visualization.heuristics_net.visualizer.get_graph(heu_net)
    #graph = assign_colors(graph, colors = {k:f"#FF0000;{random():.2f}:#FFFFFF" for k in heu_net.activities})

    colors = get_colors(dict(zip(result_df['Activity'], result_df['Normalized Average Case Duration'])), ctype="sat", base_color="#FFFFFF")
    #print(colors)
    #png_image = graph.create_png()
    #process_model = base64.b64encode(png_image).decode('utf-8')
    process_model = colorize_net(heu_net, colors)

    return OutputModel(table=prettify_df(result_df).to_dict(orient="records"), plot=plot, process_model=process_model)

# Advanced
def activity_workload_distribution_per_resource(df):
    # get time spent in each role for every resource
    total_time_activity = df.groupby(['Resource', 'Activity'])['Duration'].sum().reset_index()

    total_time_activity = total_time_activity.rename(columns={'Duration': 'Total Time Spent on Activity (hr)'})

    # convert to hours
    total_time_activity['Total Time Spent on Activity (hr)'] = (total_time_activity['Total Time Spent on Activity (hr)'].dt.total_seconds() / 3600).round(2)

    #get the overall time spent for each resource using previously defined function
    total_time_resource = worked_hours_per_resource(df)

    # Merge dfs
    result_df = pd.merge(total_time_activity, total_time_resource, on='Resource')
    
    # calculate relative time
    result_df['Percentage (%)'] = ((result_df['Total Time Spent on Activity (hr)'] / result_df['Total Time Worked (hr)']) * 100).round(2)
    
    # drop irrelevant columns
    result_df.drop(columns=['Total Time Worked', 'Total Time Worked (hr)'], inplace=True)
    
    return result_df

def capacity_resource_activity(df):
    workload = activity_workload_distribution_per_resource(df)
    availability = calculate_total_working_hours_per_resource(df)

    result_df = pd.merge(workload, availability, on='Resource', how='inner')

    result_df['Total Available Working Hours for this Activity']= ((result_df['Percentage (%)']/100)*result_df['Total Available Working Hours']).round(2)

     # drop irrelevant columns
    result_df.drop(columns=['Total Available Working Hours'], inplace=True)

    return result_df

def capacity_utilization_activity_new(df):
    activity_duration = df.groupby('Activity')['Duration'].sum().reset_index()

    activity_duration['Duration'] = (activity_duration['Duration'].dt.total_seconds() / 3600).round(2)

    activity_duration = activity_duration.rename(columns={'Duration': 'Total Time Spent (hr)'})

    activity_capacity = capacity_resource_activity(df).groupby('Activity')['Total Available Working Hours for this Activity'].sum().reset_index()

    result_df = pd.merge(activity_duration, activity_capacity, on='Activity', how='inner')

    # Calculate the capacity utilization for each activity
    result_df['Capacity Utilization (%)'] = (result_df['Total Time Spent (hr)'] / result_df['Total Available Working Hours for this Activity'] * 100).round(2)

    #return result_df
    capacity_utilization_activity_plot = result_df.copy()
    fig = px.bar(capacity_utilization_activity_plot, x='Capacity Utilization (%)', y='Activity',
        title='Percentage of Time Spent in Each Role by Resource',
        color_discrete_sequence=['#2066a8'],
        orientation="h")

    fig.update_traces(hovertemplate='Capacity Utilization: %{x}%')

    fig.update_layout(
        #width=1200,  # Width in pixels
        height=576,  # Height in pixels
        plot_bgcolor='white',
        xaxis=dict(dtick=10),  
        title={
            'y':0.9, #-0.15, #0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    plot = fig.to_json()

    heu_net = pm4py.discover_heuristics_net(df, dependency_threshold=0.99, case_id_key='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')

    colors = get_colors(dict(zip(result_df['Activity'], result_df['Capacity Utilization (%)'].fillna(0) / 100)))
    process_model = colorize_net(heu_net, colors)

    return OutputModel(table=prettify_df(result_df).replace({np.nan: None}).to_dict(orient="records"), plot=plot, process_model=process_model)


def get_color_option_1(df):
    grouped_activities = df.groupby('Activity')

    unique_activities = []
    average_duration_per_case = []

    for activity, activity_df in grouped_activities:

        case_durations = activity_df.groupby('Case ID')['Duration'].sum()

        average_duration = case_durations.mean()

        unique_activities.append(activity)
        average_duration_per_case.append(average_duration)

    result_df = pd.DataFrame({
        "Activity": unique_activities,
        "Average Case Duration": average_duration_per_case
    })
    
    result_df['Normalized Average Case Duration'] = result_df['Average Case Duration'].apply(
        lambda x: (x - result_df['Average Case Duration'].min()) / (result_df['Average Case Duration'].max() - result_df['Average Case Duration'].min()))

    colors = get_colors(dict(zip(result_df['Activity'], result_df['Normalized Average Case Duration'])), ctype="sat", base_color="#FFFFFF")
    return colors

def get_color_option_2(df):
    # get time spent in each role for every resource
    total_time_role = df.groupby(['Resource', 'Role'])['Duration'].sum().reset_index()

    total_time_role = total_time_role.rename(columns={'Duration': 'Total Time Worked in Role (hr)'})

    # convert to hours
    total_time_role['Total Time Worked in Role (hr)'] = (total_time_role['Total Time Worked in Role (hr)'].dt.total_seconds() / 3600).round(2)

    #get the overall time spent for each resource using previously defined function
    total_time_resource = worked_hours_per_resource(df)

    # Merge dfs
    result_df = pd.merge(total_time_role, total_time_resource, on='Resource')
    
    # calculate relative time
    result_df['Percentage (%)'] = ((result_df['Total Time Worked in Role (hr)'] / result_df['Total Time Worked (hr)']) * 100).round(2)
    
    # drop irrelevant columns
    result_df.drop(columns=['Total Time Worked', 'Total Time Worked (hr)'], inplace=True)
    
    #return result_df
    workload_distribution_per_resource_plot = result_df.copy()
    unique_roles = workload_distribution_per_resource_plot['Role'].unique()
    color_palette = custom_palette_2 #sns.color_palette("husl", len(unique_roles)).as_hex()

    role_color_mapping = {role: color_palette[i] for i, role in enumerate(unique_roles)}

    activity_role_df = pd.DataFrame(activities_per_role(df).table)
    unique_activities = activity_role_df['Activity'].unique()
    activity_color_mapping = {activity: split_equal([role_color_mapping[role[1]] for role in activity_role_df.loc[activity_role_df['Activity']==activity, 'Role'].items()]) for activity in unique_activities}
    
    return activity_color_mapping

def get_color_option_3(df):
        # Group by Resource and Activity, sum durations to get the total time spent on each activity by each resource
    result_df = df.groupby(['Resource', 'Activity'])['Duration'].sum().reset_index()

    # Rename columns for better readability
    result_df=result_df.rename(columns={'Duration': 'Total Time Spent'})

    # Add a new column for the overall time spent in minutes
    result_df['Total Time Spent (min)'] = (result_df['Total Time Spent'].dt.total_seconds() / 60).round(2)  # Convert to minutes

    # Calculate the total time spent per resource on all activities
    total_time_per_resource = result_df.groupby('Resource')['Total Time Spent'].sum().reset_index()
    total_time_per_resource = total_time_per_resource.rename(columns={'Total Time Spent': 'Overall Time Spent'})

    # Merge the total time spent per resource into the result dataframe
    result_df = result_df.merge(total_time_per_resource, on='Resource')

    # Calculate the percentage of time spent on each activity
    result_df['Percentage Time Spent (%)'] = (result_df['Total Time Spent'] / result_df['Overall Time Spent'] * 100).round(2)

    # Drop the 'Overall Time Spent' column to hide it
    result_df = result_df.drop(columns=['Overall Time Spent'])

    #return result_df
    resource_time_distribution_plot = result_df.copy()
    unique_activities = resource_time_distribution_plot['Activity'].unique()
    color_palette = custom_palette_1 #sns.color_palette("husl", len(unique_activities)).as_hex()

    # dictionary for dfg color coding
    activity_color_mapping = {activity: color_palette[i] for i, activity in enumerate(unique_activities)}

    return activity_color_mapping

def get_color_option_4(df):
        # Group by Role and Activity, sum durations to get the total time spent on each activity by each role
    result_df = df.groupby(['Role', 'Activity'])['Duration'].sum().reset_index()

    # Rename columns for better readability
    result_df = result_df.rename(columns={'Duration': 'Total Time Spent'})

    # Add a new column for the overall time spent in minutes
    result_df['Total Time Spent (min)'] = (result_df['Total Time Spent'].dt.total_seconds() / 60).round(2)  # Convert to minutes

    # Calculate the total time spent per resource on all activities
    total_time_per_role = result_df.groupby('Role')['Total Time Spent'].sum().reset_index()
    total_time_per_role = total_time_per_role.rename(columns={'Total Time Spent': 'Overall Time Spent'})

    # Merge the total time spent per resource into the result dataframe
    result_df = result_df.merge(total_time_per_role, on='Role')

    # Calculate the percentage of time spent on each activity
    result_df['Percentage Time Spent (%)'] = (result_df['Total Time Spent'] / result_df['Overall Time Spent'] * 100).round(2)

    # Drop the 'Overall Time Spent' column to hide it
    result_df = result_df.drop(columns=['Overall Time Spent'])

    #return result_df
    role_time_distribution_plot = result_df.copy()
    unique_activities = role_time_distribution_plot['Activity'].unique()
    color_palette = custom_palette_1 #sns.color_palette("husl", len(unique_activities)).as_hex()

    # dictionary for dfg color coding
    activity_color_mapping = {activity: color_palette[i] for i, activity in enumerate(unique_activities)}
    
    return activity_color_mapping

def get_color_option_5(df):
    activity_duration = df.groupby('Activity')['Duration'].sum().reset_index()

    activity_duration['Duration'] = (activity_duration['Duration'].dt.total_seconds() / 3600).round(2)

    activity_duration = activity_duration.rename(columns={'Duration': 'Total Time Spent (hr)'})

    activity_capacity = capacity_resource_activity(df).groupby('Activity')['Total Available Working Hours for this Activity'].sum().reset_index()

    result_df = pd.merge(activity_duration, activity_capacity, on='Activity', how='inner')

    # Calculate the capacity utilization for each activity
    result_df['Capacity Utilization (%)'] = (result_df['Total Time Spent (hr)'] / result_df['Total Available Working Hours for this Activity'] * 100).round(2)

    result_df['Capacity Utilization (%)'] = (
    result_df['Total Time Spent (hr)'] / result_df['Total Available Working Hours for this Activity'] * 100).replace([np.inf, -np.inf], np.nan).round(2)

    result_df['Capacity Utilization (%)'] = result_df['Capacity Utilization (%)'].fillna(0)  # or another default

    return dict(zip(result_df['Activity'], result_df['Capacity Utilization (%)']))